# Copyright (c) 2023-2024, ETH Zurich (Robotics Systems Lab)
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tiny stdlib-only HTTP server exposing mission_vlm.VLMPolicy to processes that can't import it
directly -- specifically, sar_demo.py running inside the isaac-lab-base-viplanner Docker
container's own bundled Python, which can't share a process with this repo's `mission_vlm` conda
env (see MISSION_VLM_MODULE.md's design-decisions section for why: transformers-new-enough-for-
Qwen2-VL needs huggingface-hub>=0.23, which conflicts with the huggingface-hub==0.13.4 NoMaD's
diffusers==0.11.1 pin requires in every other env this repo uses).

The container uses `network_mode: host` (see ~/IsaacLab-viplanner/docker/docker-compose.yaml), so
a server bound to 127.0.0.1 on the host is directly reachable from inside the container via
localhost -- no port mapping needed.

Run with: `python viplanner/utils/mission_vlm_server.py` in the `mission_vlm` conda env. Loads the
model once at startup (not per-request) so callers don't pay the ~14.5s load time on every call.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

sys.path.insert(0, "/home/ashain/Desktop/viplanner")
from viplanner.utils.mission_vlm import MissionSpec, VLMPolicy

_policy: VLMPolicy | None = None


def _decode_png_b64(data: str) -> np.ndarray:
    from PIL import Image

    return np.array(Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB"))


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send_json(200, {"status": "ok", "model_id": _policy.model_id if _policy else None})
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        try:
            request = json.loads(self.rfile.read(length))
        except json.JSONDecodeError as e:
            self._send_json(400, {"error": f"invalid JSON body: {e}"})
            return

        try:
            if self.path == "/parse_mission":
                mission = _policy.parse_mission(request["instruction"])
                self._send_json(200, {
                    "target_object": mission.target_object,
                    "target_region_keywords": mission.target_region_keywords,
                })
            elif self.path == "/next_semantic_subgoal":
                rgb = _decode_png_b64(request["rgb_png_b64"])
                mission = MissionSpec(
                    raw_instruction="",
                    target_object=request["target_object"],
                    target_region_keywords=request["target_region_keywords"],
                )
                subgoal = _policy.next_semantic_subgoal(
                    rgb, mission, request["cam_heading_rad"], request.get("cam_fov_rad", 1.02)
                )
                self._send_json(200, {
                    "heading_bias_rad": subgoal.heading_bias_rad,
                    "confidence": subgoal.confidence,
                    "rationale": subgoal.rationale,
                    "victim_hint": subgoal.victim_hint,
                })
            else:
                self._send_json(404, {"error": "not found"})
        except Exception as e:  # noqa: BLE001 -- this is a request-handling boundary, must not crash the server
            self._send_json(500, {"error": f"{type(e).__name__}: {e}"})

    def log_message(self, fmt, *args) -> None:  # quieter default logging, one line per request
        print(f"[mission_vlm_server] {self.address_string()} {fmt % args}")


def main() -> None:
    global _policy
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    print("Loading VLMPolicy (first call downloads/loads the model)...")
    _policy = VLMPolicy()
    _policy._ensure_loaded()
    print(f"Model loaded on {_policy._device}. Serving on http://{args.host}:{args.port}")

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.serve_forever()


if __name__ == "__main__":
    main()
