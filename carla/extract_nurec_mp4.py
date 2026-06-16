"""
從 38 個未使用的 NuRec 場景 MP4 抽幀，存到 datasets/nurec_raw/
每幀儲存為 PNG，命名格式：UUID_XXXXXX.png（與現有 nurec/ 一致）

Usage:
  conda run -n carla_env python3 extract_nurec_mp4.py
"""
import cv2
import subprocess
from pathlib import Path

USDZ_ROOT = Path("/home/itriu100/PhysicalAI-Autonomous-Vehicles-NuRec/sample_set/26.02_release")
OUTPUT_DIR = Path("/home/itriu100/carla/datasets/nurec_raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 已使用的 13 個場景（跳過）
USED_SCENES = {
    "01d503d4-449b-46fc-8d78-9085e70d3554",
    "0245ff75-aa3f-46b7-ba87-16a7afb841af",
    "026d6a39-bd8f-4175-bc61-fe50ed0403a3",
    "04749bb9-9b37-495b-bed0-77f0e33ac7da",
    "05bb8212-63e1-40a8-b4fc-3142c0e94646",
    "05ecbbf5-d88c-41b7-97f2-f4ac2615a8c9",
    "05f35348-9de0-4f68-ac65-f31316dbb59e",
    "060131e7-ee72-477c-89da-083e0b446566",
    "065dcac9-ee67-4434-a835-c6b816c88e48",
    "07981e6a-22dd-4796-ad2f-1252037ecd28",
    "07ab4a5b-e934-4fff-b5d1-47efedadac47",
    "09a95ffa-37b7-486d-9404-ac8cf1c3e045",
    "0a18c5a4-9aca-4efd-b604-c75f3269c502",
}


def get_frame_count(mp4_path: Path) -> int:
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "stream=nb_frames",
         "-of", "csv=p=0", str(mp4_path)],
        capture_output=True, text=True
    )
    try:
        return int(result.stdout.strip().split("\n")[0])
    except Exception:
        return 0


def extract_scene(scene_id: str, mp4_path: Path, out_dir: Path) -> int:
    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        print(f"  [ERROR] Cannot open {mp4_path}")
        return 0

    saved = 0
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out_path = out_dir / f"{scene_id}_{frame_idx:06d}.png"
        if not out_path.exists():
            cv2.imwrite(str(out_path), frame)
        saved += 1
        frame_idx += 1

    cap.release()
    return saved


def main():
    scenes = sorted([
        d for d in USDZ_ROOT.iterdir()
        if d.is_dir() and d.name not in USED_SCENES
        and (d / "camera_front_wide_120fov.mp4").exists()
    ])

    print(f"Found {len(scenes)} new scenes to extract")
    total = 0
    for i, scene_dir in enumerate(scenes):
        scene_id = scene_dir.name
        mp4 = scene_dir / "camera_front_wide_120fov.mp4"
        n = get_frame_count(mp4)
        print(f"[{i+1:02d}/{len(scenes)}] {scene_id}  ({n} frames)")
        saved = extract_scene(scene_id, mp4, OUTPUT_DIR)
        total += saved
        print(f"  → saved {saved}")

    print(f"\nDone. Total extracted: {total} frames → {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
