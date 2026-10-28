"""Mask2Former inference server for the sim's control loop (M2F=1 in run_stair_usd.sh).

Runs on the host in the `mask2former_env` conda env; the Isaac container (--network=host) posts the
camera's colour image each replan tick and gets back the ViPlanner semantic colour image:
    POST /seg   headers X-H, X-W   body: H*W*3 uint8 RGB   ->   body: H*W*3 uint8 RGB (ViPlanner colours)
Pixels Mask2Former leaves unlabelled, and COCO classes with no ViPlanner mapping, become "static"
(an obstacle). Start:  python m2f_server.py [--model ...] [--port 8765]
"""
import argparse
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

from m2f_test import VIP_COLOR, build_class_table, segment

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="facebook/mask2former-swin-large-coco-panoptic")
ap.add_argument("--port", type=int, default=8765)
args = ap.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
proc = AutoImageProcessor.from_pretrained(args.model)
model = Mask2FormerForUniversalSegmentation.from_pretrained(args.model).to(device).eval()
table = build_class_table({int(k): v for k, v in model.config.id2label.items()})
# colour lookup: index 0 = unlabelled, index cid+1 = model class cid
lut = np.zeros((max(table) + 2, 3), dtype=np.uint8)
lut[0] = VIP_COLOR["static"]
for cid, name in table.items():
    lut[cid + 1] = VIP_COLOR.get(name, VIP_COLOR["static"])
stats = {"n": 0, "t": 0.0}


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        h, w = int(self.headers["X-H"]), int(self.headers["X-W"])
        body = self.rfile.read(int(self.headers["Content-Length"]))
        rgb = np.frombuffer(body, dtype=np.uint8).reshape(h, w, 3)
        t0 = time.time()
        cls = segment(model, proc, Image.fromarray(rgb), device)
        out = lut[cls + 1].tobytes()
        stats["n"] += 1
        stats["t"] += time.time() - t0
        if stats["n"] % 200 == 0:
            print(f"[m2f_server] {stats['n']} images, mean {1000 * stats['t'] / stats['n']:.0f} ms", flush=True)
        self.send_response(200)
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a):
        pass


print(f"[m2f_server] {args.model} on {device}, listening on 127.0.0.1:{args.port}", flush=True)
HTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
