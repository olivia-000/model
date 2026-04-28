"""
LLM 串接 Stage2 worker for expansion joint anomaly detection.

Architecture:
- Per-channel Queue (unlimited by default) + per-channel worker thread
- Gemini / Qwen-local → OpenAI-compatible API (openai SDK + base_url switching)
- Claude              → anthropic SDK (no full OpenAI-compatible layer)
- Emits add_log_signal after each result; saves annotated image to disk

Note: LLM results are async (1-5s latency). They are never drawn on the live
video feed; results appear only in GUI log and saved annotated images.
"""

import re
import json
import time
import queue
import base64
import threading
from datetime import datetime

import cv2
import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal
from ultralytics.utils.plotting import Annotator


# Provider → available models (displayed in GUI dropdown)
# gemini / qwen_local → OpenAI-compatible API (openai SDK)
# claude              → anthropic SDK
PROVIDER_MODELS = {
    'gemini': [
        'gemini-2.5-flash',
        'gemini-2.5-flash-lite',
        'gemini-2.0-flash',
        'gemini-2.0-flash-lite',
    ],
    'claude': [
        'claude-haiku-4-5-20251001',
        'claude-3-5-haiku-20241022',
    ],
    'qwen_local': [
        'Qwen/Qwen2-VL-7B-Instruct-AWQ',
        'Qwen2.5-VL-7B-Instruct',
        'Qwen2.5-VL-72B-Instruct',
        'Qwen2-VL-7B-Instruct',
    ],
}

# OpenAI-compatible base URLs
_OPENAI_BASE_URLS = {
    'gemini': 'https://generativelanguage.googleapis.com/v1beta/openai/',
    # qwen_local uses cfg qwen_base_url (configurable)
}

VLM_PROMPT = (
    "這張圖片顯示橋樑伸縮縫的近景。請仔細判斷：\n"
    "1. 是否有高低差異常？（伸縮縫兩側鋼板高度不平，有明顯落差）\n"
    "2. 是否有阻塞異常？（縫隙被異物、泥土或雜物堵塞）\n"
    "只回覆 JSON，不要任何其他文字或 markdown：\n"
    "{\"height_diff\": true, \"blocked\": false}"
)


class VLMWorker(QObject):
    """Async LLM Stage2 worker. One shared instance per Detection model."""

    add_log_signal = pyqtSignal(str)

    def __init__(self, cfg_dict: dict) -> None:
        super().__init__()
        self._is_running = False
        self._started = False          # guard against double-start
        self._cfg_setup(cfg_dict)
        self._build_queues()
        self._threads: dict[str, threading.Thread] = {}

    # ── Configuration ────────────────────────────────────────────────

    def _cfg_setup(self, cfg_dict: dict) -> None:
        self.cfg = cfg_dict
        vlm = cfg_dict.get('vlm', {})
        self.provider    = vlm.get('provider', 'gemini')
        self.model_name  = vlm.get('model', 'gemini-2.5-flash')
        self.api_key_gemini  = vlm.get('api_key_gemini', '')
        self.api_key_claude  = vlm.get('api_key_claude', '')
        self.qwen_base_url   = vlm.get('qwen_base_url', 'http://localhost:8000/v1')
        self.vlm_channels    = vlm.get('vlm_channels', ['ch1', 'ch3', 'ch4'])
        self.queue_max_size  = vlm.get('queue_max_size', 0)   # 0 = unlimited
        self.cooldown_sec    = vlm.get('cooldown_sec', 30)    # min seconds between submissions per channel
        self.result_img_path = cfg_dict.get('result_img_path', 'detection_result_image_data')

    def _build_queues(self) -> None:
        maxsize = self.queue_max_size if self.queue_max_size > 0 else 0
        self._queues: dict[str, queue.Queue] = {
            ch: queue.Queue(maxsize=maxsize) for ch in self.vlm_channels
        }
        self._last_submit_time: dict[str, float] = {}  # cooldown tracker per channel

    # ── Lifecycle ────────────────────────────────────────────────────

    def start(self) -> None:
        if self._started:
            return  # idempotent — prevent duplicate threads on re-enable
        self._started = True
        self._is_running = True
        for ch in self.vlm_channels:
            t = threading.Thread(
                target=self._worker_loop, args=(ch,),
                name=f'vlm_{ch}', daemon=True
            )
            t.start()
            self._threads[ch] = t
        print(f"[LLM] Worker started | provider={self.provider} model={self.model_name} "
              f"channels={self.vlm_channels}")

    def stop(self) -> None:
        self._is_running = False

    # ── Job submission ───────────────────────────────────────────────

    def submit(
        self,
        ch_key: str,
        crop: np.ndarray,
        annotated_frame: np.ndarray,
        box_xyxy,
        stage1_label: str,
        timestamp: datetime,
        location: str,
    ) -> None:
        """Submit a LLM inference job. Returns immediately (non-blocking)."""
        if ch_key not in self._queues:
            return
        # Cooldown: skip if last submission for this channel was too recent
        now = time.time()
        if self.cooldown_sec > 0 and now - self._last_submit_time.get(ch_key, 0) < self.cooldown_sec:
            return
        self._last_submit_time[ch_key] = now
        q = self._queues[ch_key]
        # If queue is full (max_size > 0), drop oldest to make room
        if self.queue_max_size > 0 and q.full():
            try:
                q.get_nowait()
            except queue.Empty:
                pass
        job = {
            'ch_key':          ch_key,
            'crop':            crop.copy(),
            'annotated_frame': annotated_frame.copy(),
            'box_xyxy':        box_xyxy,
            'stage1_label':    stage1_label,
            'timestamp':       timestamp,
            'location':        location,
        }
        q.put(job)

    # ── Worker loop ──────────────────────────────────────────────────

    def _worker_loop(self, ch_key: str) -> None:
        q = self._queues[ch_key]
        while self._is_running:
            try:
                job = q.get(timeout=1.0)
            except queue.Empty:
                continue
            self._process_job(job)

    def _process_job(self, job: dict) -> None:
        ch_key         = job['ch_key']
        crop           = job['crop']
        annotated_frame = job['annotated_frame']
        box_xyxy       = job['box_xyxy']
        stage1_label   = job['stage1_label']
        timestamp: datetime = job['timestamp']
        location       = job['location']

        timestamp_str = timestamp.strftime('%Y-%m-%d_%H-%M-%S')
        t0 = time.time()

        try:
            result = self._call_vlm(crop)
        except Exception as exc:
            elapsed = time.time() - t0
            print(f"[LLM] {ch_key} API error ({elapsed:.1f}s): {exc}")
            log_msg = (f"{timestamp_str} | {ch_key} | {location} | "
                       f"[LLM] error: {exc}")
            self.add_log_signal.emit(log_msg)
            return

        elapsed = time.time() - t0
        height_diff = result.get('height_diff', False)
        blocked     = result.get('blocked', False)
        is_anomaly  = height_diff or blocked

        # Build human-readable label
        parts = []
        if height_diff:
            parts.append('height diff')
        if blocked:
            parts.append('blocked')
        vlm_label = ' + '.join(parts) if parts else 'normal'

        full_label = f"{stage1_label} [{vlm_label}] ({elapsed:.1f}s)"
        color = (0, 0, 255) if is_anomaly else (0, 255, 0)

        # Draw VLM result on the saved-frame copy and overwrite to disk
        annotator = Annotator(annotated_frame, line_width=5, font_size=10)
        annotator.box_label(box_xyxy, label=full_label, color=color)

        self._save_annotated(annotated_frame, timestamp_str, ch_key)

        # Emit log entry (shows up in GUI log list)
        log_msg = (f"{timestamp_str} | {ch_key} | {location} | "
                   f"[LLM/{self.provider}] {vlm_label} ({elapsed:.1f}s)")
        self.add_log_signal.emit(log_msg)

        # Debug print
        print(f"[LLM] {ch_key} | {vlm_label} | {elapsed:.1f}s | "
              f"raw={result}")

    # ── Image save ───────────────────────────────────────────────────

    def _save_annotated(self, frame: np.ndarray, timestamp_str: str, ch_key: str) -> None:
        from utils import save_image
        filename = f"{timestamp_str}_{ch_key}.jpeg"
        save_image(frame, filename, ch_key, f"{self.result_img_path}/annotated")

    # ── API calls ────────────────────────────────────────────────────

    def _call_vlm(self, crop: np.ndarray) -> dict:
        if self.provider in ('gemini', 'qwen_local'):
            return self._call_openai_compatible(crop)
        elif self.provider == 'claude':
            return self._call_claude(crop)
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")

    def _call_openai_compatible(self, crop: np.ndarray) -> dict:
        """Gemini and Qwen-local both use OpenAI SDK with different base_url."""
        from openai import OpenAI

        if self.provider == 'gemini':
            base_url = _OPENAI_BASE_URLS['gemini']
            api_key  = self.api_key_gemini
        else:  # qwen_local
            base_url = self.qwen_base_url
            api_key  = 'EMPTY'  # vLLM local server doesn't require a real key

        client = OpenAI(api_key=api_key, base_url=base_url)
        _, buf = cv2.imencode('.jpeg', crop)
        b64 = base64.b64encode(buf.tobytes()).decode('utf-8')

        response = client.chat.completions.create(
            model=self.model_name,
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    },
                    {"type": "text", "text": VLM_PROMPT},
                ],
            }],
        )
        return self._parse_json(response.choices[0].message.content.strip())

    def _call_claude(self, crop: np.ndarray) -> dict:
        """Claude uses anthropic SDK — no full OpenAI-compatible layer available."""
        import anthropic

        client = anthropic.Anthropic(api_key=self.api_key_claude)
        _, buf = cv2.imencode('.jpeg', crop)
        b64 = base64.b64encode(buf.tobytes()).decode('utf-8')

        message = client.messages.create(
            model=self.model_name,
            max_tokens=100,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": b64,
                        },
                    },
                    {"type": "text", "text": VLM_PROMPT},
                ],
            }],
        )
        return self._parse_json(message.content[0].text.strip())

    def _parse_json(self, text: str) -> dict:
        """Strip markdown fences if present, then parse JSON."""
        text = re.sub(r'```(?:json)?\s*', '', text).strip().rstrip('`').strip()
        return json.loads(text)

    # ── Runtime config update (called from GUI apply) ────────────────

    def update_provider_model(self, provider: str, model_name: str) -> None:
        self.provider   = provider
        self.model_name = model_name
        print(f"[LLM] Updated → provider={provider} model={model_name}")

    def update_cooldown(self, cooldown_sec: int) -> None:
        self.cooldown_sec = cooldown_sec
        print(f"[LLM] cooldown_sec={cooldown_sec}")
