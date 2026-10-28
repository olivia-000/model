# Copyright (c) 2023-2024, ETH Zurich (Robotics Systems Lab)
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Mission-understanding (VLM/VLN) layer: turns a natural-language SAR instruction (e.g. "go
search the collapsed area ahead for trapped victims") into a semantic subgoal that biases which
frontier the frontier module (frontier_map.py) picks -- the top of the hierarchy in
NOMAD_INTEGRATION.md, sitting above NoMaD/frontier without changing either's interface.

Deliberately mission-cadence, not per-step: `parse_mission` runs once at mission start,
`next_semantic_subgoal` runs only when `should_reground_mission` says enough time has passed (see
that function's docstring for why this is time-gated rather than distance-gated like NoMaD's own
`nomad_bridge.should_replan_subgoal`). A local 2B-class VLM is inherently seconds-per-call, not
milliseconds -- fine at this cadence, wrong at ViPlanner's ~150Hz or even NoMaD's per-subgoal rate.

Runs in its own `mission_vlm` conda env, not the `viplanner` env NoMaD/ViPlanner use: any
transformers version new enough to support Qwen2-VL needs huggingface-hub>=0.23, which conflicts
with the huggingface-hub==0.13.4 the `viplanner` env's diffusers==0.11.1 (NoMaD) is pinned to (see
NOMAD_INTEGRATION.md's "known gotcha" -- newer hub/diffusers breaks NoMaD's imports). Same
isolation pattern this machine already uses for mask2former_env/ros_env. This module has no other
dependency on the rest of the `viplanner` package, so it's fully usable standalone from that env.

The `*_remote` functions at the bottom of this file are the other side of that isolation: thin
HTTP clients for `mission_vlm_server.py` (which runs `VLMPolicy` in the `mission_vlm` env and
exposes it over `network_mode: host`-reachable localhost HTTP), safe to import from a THIRD
Python environment that can't have transformers/torch at all -- e.g. sar_demo.py running inside
the isaac-lab-base-viplanner container's own bundled Python. They only need `json`/`urllib`/
`base64`/`numpy`/`PIL`, all already imported at module level above (or stdlib), never touching
`VLMPolicy`'s deferred torch/transformers imports.
"""

from __future__ import annotations

import base64
import io
import json
import re
import urllib.request
from dataclasses import dataclass, field

import numpy as np

DEFAULT_MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"

# Coarse verbal directions a VQA-style small VLM can reliably distinguish in one image -- ordered
# left-to-right, symmetric around "center". Fractions of the camera's half-FOV; "unclear" isn't
# listed here, it maps to heading_bias_rad=None in _bucket_to_angle.
_DIRECTION_BUCKET_FRACTIONS = {
    "left": -0.8,
    "center_left": -0.4,
    "center": 0.0,
    "center_right": 0.4,
    "right": 0.8,
}
# Longest/most-specific keys first so substring matching below doesn't match "left" inside
# "center_left" (or "center-left") before the more specific bucket gets a chance.
_DIRECTION_BUCKET_ORDER = ("center_left", "center_right", "left", "right", "center")

_VICTIM_KEYWORDS = ("person", "people", "victim", "survivor", "human", "casualty")


@dataclass
class MissionSpec:
    """Structured mission understanding parsed once from the raw human instruction."""

    raw_instruction: str
    target_object: str
    target_region_keywords: list[str] = field(default_factory=list)


@dataclass
class SemanticSubgoal:
    """One VLM-grounded exploration hint, re-derived at mission cadence (see module docstring).

    `heading_bias_rad` is world-frame; feed straight into
    `frontier_map.OnlineOccupancyGrid.select_frontier(..., heading_bias_rad=...)`.
    `heading_bias_rad=None` means the VLM found nothing directional in view -- caller should fall
    back to pure-geometric frontier selection (select_frontier's existing default behavior).
    `victim_hint` is a soft signal only, not a detection -- `victim_detection.detect_persons`
    (YOLO) stays the authoritative source for actually driving toward a person.
    """

    heading_bias_rad: float | None
    confidence: float
    rationale: str
    victim_hint: bool


def _bucket_to_angle(bucket: str, cam_heading_rad: float, cam_fov_rad: float) -> float | None:
    """Pure conversion, no model involved -- kept separate from next_semantic_subgoal so the
    bucket->angle mapping is directly unit-testable without loading a VLM."""
    fraction = _DIRECTION_BUCKET_FRACTIONS.get(bucket)
    if fraction is None:
        return None
    return cam_heading_rad + fraction * (cam_fov_rad / 2.0)


def _extract_bucket(text: str) -> str | None:
    """Find the first known direction bucket mentioned in a free-form VLM response. Checks
    multi-word buckets before single-word ones (see _DIRECTION_BUCKET_ORDER) so "center-left"
    isn't mis-matched as bare "left"."""
    normalized = text.lower().replace("-", "_").replace(" ", "_")
    for bucket in _DIRECTION_BUCKET_ORDER:
        if bucket in normalized:
            return bucket
    return None


def _parse_mission_json(raw_instruction: str, response_text: str) -> MissionSpec:
    """Permissive parse of the VLM's mission-parse response: try strict JSON first, fall back to
    regex extraction, and never raise -- a slightly malformed response from a 2B model shouldn't
    hard-fail the whole mission."""
    try:
        # small local models often wrap JSON in prose or code fences; grab the first {...} block
        match = re.search(r"\{.*\}", response_text, re.DOTALL)
        payload = json.loads(match.group(0) if match else response_text)
        target_object = str(payload.get("target_object", "person"))
        keywords = payload.get("target_region_keywords", [])
        if isinstance(keywords, str):
            keywords = [k.strip() for k in keywords.split(",") if k.strip()]
        return MissionSpec(raw_instruction, target_object, [str(k) for k in keywords])
    except (json.JSONDecodeError, AttributeError, TypeError):
        pass

    obj_match = re.search(r'"?target_object"?\s*[:=]\s*"?([\w\s]+?)["\n,}]', response_text, re.IGNORECASE)
    target_object = obj_match.group(1).strip() if obj_match else "person"
    keywords = re.findall(r'"([^"]+)"', response_text)
    keywords = [k for k in keywords if k.lower() != target_object.lower()]
    return MissionSpec(raw_instruction, target_object, keywords)


_MISSION_PROMPT_TEMPLATE = """You are the mission-understanding layer of a search-and-rescue \
robot. A human operator gave this instruction: "{instruction}"

Extract what the robot should search FOR and what kind of AREA/terrain it should look for, as \
JSON only, no other text:
{{"target_object": "<what to find, e.g. person, victim>", \
"target_region_keywords": ["<short area/terrain descriptors, e.g. collapsed, rubble, corridor>"]}}"""

_SUBGOAL_PROMPT_TEMPLATE = """You are the mission-understanding layer of a search-and-rescue \
robot searching for: {target_object}, in an area matching: {region_keywords}.

Look at this image. Answer as JSON only, no other text:
{{"direction": "<one of: left, center_left, center, center_right, right, unclear>", \
"victim_visible": <true or false>, "rationale": "<one short sentence>"}}"""


class VLMPolicy:
    """Loads a local VQA-capable VLM and answers the two mission-understanding questions this
    layer needs: what does the instruction mean (`parse_mission`), and given the current view,
    which way looks like it matches (`next_semantic_subgoal`). See module docstring for why this
    runs in its own conda env."""

    def __init__(self, model_id: str = DEFAULT_MODEL_ID, device: str | None = None):
        self.model_id = model_id
        self._device = device
        self._model = None
        self._processor = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_id, torch_dtype=torch.float16 if device == "cuda" else torch.float32
        ).to(device).eval()
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._device = device

    def _generate(self, prompt: str, image: np.ndarray | None) -> str:
        import torch
        from PIL import Image

        self._ensure_loaded()
        content = []
        if image is not None:
            content.append({"type": "image", "image": Image.fromarray(image)})
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]

        text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images = [Image.fromarray(image)] if image is not None else None
        inputs = self._processor(text=[text], images=images, return_tensors="pt").to(self._device)

        with torch.no_grad():
            output_ids = self._model.generate(**inputs, max_new_tokens=128, do_sample=False)
        generated = output_ids[:, inputs["input_ids"].shape[1] :]
        return self._processor.batch_decode(generated, skip_special_tokens=True)[0]

    def parse_mission(self, instruction: str) -> MissionSpec:
        """One text-only call at mission start -- extracts what to search for and what the target
        area/terrain looks like from the raw human instruction."""
        response = self._generate(_MISSION_PROMPT_TEMPLATE.format(instruction=instruction), image=None)
        return _parse_mission_json(instruction, response)

    def next_semantic_subgoal(
        self,
        rgb_image: np.ndarray,
        mission: MissionSpec,
        cam_heading_rad: float,
        cam_fov_rad: float = 1.02,
    ) -> SemanticSubgoal:
        """VQA-style call: given the mission's target region keywords, which coarse direction in
        this image looks like it leads there. `cam_fov_rad` should be the RGB camera's actual
        horizontal FOV when available (default ~58deg is a placeholder, not measured from this
        repo's camera configs)."""
        prompt = _SUBGOAL_PROMPT_TEMPLATE.format(
            target_object=mission.target_object,
            region_keywords=", ".join(mission.target_region_keywords) or "unspecified",
        )
        response = self._generate(prompt, image=rgb_image)

        try:
            match = re.search(r"\{.*\}", response, re.DOTALL)
            payload = json.loads(match.group(0) if match else response)
            bucket = str(payload.get("direction", "unclear")).lower()
            victim_visible = bool(payload.get("victim_visible", False))
            rationale = str(payload.get("rationale", response))[:200]
        except (json.JSONDecodeError, AttributeError, TypeError):
            bucket = _extract_bucket(response) or "unclear"
            victim_visible = any(kw in response.lower() for kw in _VICTIM_KEYWORDS)
            rationale = response[:200]

        heading_bias_rad = _bucket_to_angle(bucket, cam_heading_rad, cam_fov_rad)
        confidence = 0.6 if heading_bias_rad is not None else 0.0
        return SemanticSubgoal(heading_bias_rad, confidence, rationale, victim_visible)


def should_reground_mission(last_reground_time: float, now: float, min_interval_s: float = 15.0) -> bool:
    """Time-gate for calling next_semantic_subgoal again. Mission-level regrounding isn't tied to
    reaching one specific waypoint the way NoMaD subgoals are (nomad_bridge.should_replan_subgoal
    is distance-based for exactly that reason) -- it's "has enough time passed to re-ask the VLM
    given a multi-second call cost", so a plain wall-clock gate is enough; no model needed to
    evaluate this."""
    return (now - last_reground_time) >= min_interval_s


def _http_post_json(url: str, payload: dict, timeout: float) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def parse_mission_remote(base_url: str, instruction: str, timeout: float = 30.0) -> MissionSpec:
    """Client for mission_vlm_server.py's /parse_mission -- see module docstring for why this is a
    separate HTTP call rather than a direct VLMPolicy call. Usable from any Python environment with
    only stdlib + numpy (no torch/transformers needed)."""
    response = _http_post_json(f"{base_url}/parse_mission", {"instruction": instruction}, timeout)
    return MissionSpec(instruction, response["target_object"], response["target_region_keywords"])


def next_semantic_subgoal_remote(
    base_url: str,
    rgb_image: np.ndarray,
    mission: MissionSpec,
    cam_heading_rad: float,
    cam_fov_rad: float = 1.02,
    timeout: float = 30.0,
) -> SemanticSubgoal:
    """Client for mission_vlm_server.py's /next_semantic_subgoal. See parse_mission_remote."""
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(rgb_image).save(buf, format="PNG")
    rgb_png_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    response = _http_post_json(
        f"{base_url}/next_semantic_subgoal",
        {
            "rgb_png_b64": rgb_png_b64,
            "target_object": mission.target_object,
            "target_region_keywords": mission.target_region_keywords,
            "cam_heading_rad": cam_heading_rad,
            "cam_fov_rad": cam_fov_rad,
        },
        timeout,
    )
    return SemanticSubgoal(
        response["heading_bias_rad"], response["confidence"], response["rationale"], response["victim_hint"]
    )
