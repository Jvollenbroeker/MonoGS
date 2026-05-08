"""
Preprocess a ScanNet++ iPhone capture into a MonoGS-friendly layout.

Input  (raw ScanNet++ scene directory):
    <scene>/rgb.mkv                  H.264 video, 60 fps, 1920x1440
    <scene>/depth.bin                LZ4 chunks of uint16 192x256 (mm)
    <scene>/pose_intrinsic_imu.json  per-frame pose + intrinsic + IMU

Output (default: <scene>/processed/):
    rgb/frame_XXXXXX.jpg     RGB at requested scale
    depth/frame_XXXXXX.png   uint16 PNG (mm), upsampled to RGB resolution
    traj.txt                 one 4x4 row-major pose per kept frame (w2c, OpenCV)
    intrinsics.txt           fx fy cx cy width height  (post-resize)
    frame_ids.txt            original ARKit frame index for each kept frame

The output mirrors the ReplicaParser-style layout that MonoGS already
consumes, so the dataloader stays simple.

Pose convention:
    pose_intrinsic_imu.json["aligned_pose"] is camera-to-world in ARKit
    (OpenGL: x-right, y-up, z-back). We flip y/z to OpenCV camera axes,
    then invert to world-to-camera, which is what MonoGS expects.
"""

import argparse
import json
import os
import struct
from pathlib import Path

import cv2
import lz4.block
import numpy as np
from tqdm import tqdm

DEPTH_H, DEPTH_W = 192, 256
DEPTH_FRAME_BYTES = DEPTH_H * DEPTH_W * 2  # uint16
GL_TO_CV = np.diag([1.0, -1.0, -1.0, 1.0])


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("scene_dir", type=Path,
                   help="raw ScanNet++ scene dir (contains rgb.mkv, depth.bin, ...)")
    p.add_argument("--out", type=Path, default=None,
                   help="output dir (default: <scene>/processed)")
    p.add_argument("--stride", type=int, default=10,
                   help="keep every Nth frame (default 10 -> 795 frames @ 6Hz)")
    p.add_argument("--scale", type=float, default=0.5,
                   help="RGB resize factor (default 0.5 -> 960x720)")
    p.add_argument("--max-frames", type=int, default=None,
                   help="optional cap on number of kept frames")
    return p.parse_args()


def iter_depth_frames(depth_path: Path):
    """Yield (frame_idx, uint16 ndarray of shape (192,256)) from depth.bin."""
    with open(depth_path, "rb") as f:
        idx = 0
        while True:
            header = f.read(4)
            if len(header) < 4:
                return
            size = struct.unpack("<I", header)[0]
            payload = f.read(size)
            if len(payload) < size:
                raise IOError(f"truncated depth.bin at frame {idx}")
            raw = lz4.block.decompress(payload, uncompressed_size=DEPTH_FRAME_BYTES)
            yield idx, np.frombuffer(raw, dtype=np.uint16).reshape(DEPTH_H, DEPTH_W)
            idx += 1


def main():
    args = parse_args()
    scene = args.scene_dir.resolve()
    out = (args.out or scene / "processed").resolve()
    (out / "rgb").mkdir(parents=True, exist_ok=True)
    (out / "depth").mkdir(parents=True, exist_ok=True)

    # Load poses + intrinsics
    pose_json = json.load(open(scene / "pose_intrinsic_imu.json"))
    n_total = len(pose_json)
    print(f"pose_intrinsic_imu.json: {n_total} frames")

    # Pick which frame indices to keep
    keep = list(range(0, n_total, args.stride))
    if args.max_frames:
        keep = keep[: args.max_frames]
    keep_set = set(keep)
    print(f"keeping {len(keep)} frames (stride={args.stride})")

    # --- RGB extraction ---
    cap = cv2.VideoCapture(str(scene / "rgb.mkv"))
    n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if n_video != n_total:
        print(f"warning: rgb.mkv has {n_video} frames but pose json has {n_total}")
    dst_w = int(round(src_w * args.scale))
    dst_h = int(round(src_h * args.scale))
    print(f"rgb: {src_w}x{src_h} -> {dst_w}x{dst_h}")

    frame_idx = 0
    pbar = tqdm(total=len(keep), desc="rgb")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx in keep_set:
            if args.scale != 1.0:
                frame = cv2.resize(frame, (dst_w, dst_h), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(out / "rgb" / f"frame_{frame_idx:06d}.jpg"), frame,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            pbar.update(1)
        frame_idx += 1
    cap.release()
    pbar.close()

    # --- Depth extraction ---
    pbar = tqdm(total=len(keep), desc="depth")
    for idx, depth_mm in iter_depth_frames(scene / "depth.bin"):
        if idx not in keep_set:
            continue
        depth_up = cv2.resize(depth_mm, (dst_w, dst_h), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(out / "depth" / f"frame_{idx:06d}.png"), depth_up)
        pbar.update(1)
        if idx > max(keep):
            break
    pbar.close()

    # --- Poses + intrinsics ---
    # Use the FIRST kept frame's intrinsic for the calibration; per-frame
    # intrinsic variation in ARKit is tiny so a single global K is fine.
    K_full = np.array(pose_json[f"frame_{keep[0]:06d}"]["intrinsic"], dtype=np.float64)
    fx_full, fy_full = K_full[0, 0], K_full[1, 1]
    cx_full, cy_full = K_full[0, 2], K_full[1, 2]
    fx, fy = fx_full * args.scale, fy_full * args.scale
    cx, cy = cx_full * args.scale, cy_full * args.scale

    with open(out / "intrinsics.txt", "w") as f:
        f.write(f"{fx} {fy} {cx} {cy} {dst_w} {dst_h}\n")

    with open(out / "traj.txt", "w") as fpose, \
         open(out / "frame_ids.txt", "w") as fids:
        for idx in keep:
            entry = pose_json[f"frame_{idx:06d}"]
            T_c2w_gl = np.array(entry["aligned_pose"], dtype=np.float64)
            T_c2w_cv = T_c2w_gl @ GL_TO_CV
            T_w2c_cv = np.linalg.inv(T_c2w_cv)
            fpose.write(" ".join(f"{v:.10f}" for v in T_w2c_cv.flatten()) + "\n")
            fids.write(f"{idx}\n")

    print(f"\nDone. Output written to: {out}")
    print(f"  RGB+depth frames: {len(keep)}")
    print(f"  Calibration: fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}"
          f" w={dst_w} h={dst_h}")
    print(f"  Use these in your config under Dataset.Calibration.")


if __name__ == "__main__":
    main()
