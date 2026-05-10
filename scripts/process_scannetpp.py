"""
Preprocess a ScanNet++ iPhone capture into a MonoGS-friendly layout.

Uses COLMAP poses (colmap/images.txt) for reliable, unambiguous camera poses.
No coordinate-system conversion is required: COLMAP already gives world-to-camera
transforms in the standard OpenCV camera convention (x-right, y-down, z-forward),
which is exactly what MonoGS expects.

Input  (raw ScanNet++ scene directory):
    <scene>/rgb.mkv                  H.264 video, ~60 fps, 1920×1440
    <scene>/depth.bin                LZ4 chunks of uint16 192×256 (mm)
    <scene>/colmap/cameras.txt       COLMAP camera model and intrinsics
    <scene>/colmap/images.txt        COLMAP world-to-camera poses

Output (default: <scene>/processed/):
    rgb/frame_XXXXXX.jpg     RGB at requested scale (JPEG q92)
    depth/frame_XXXXXX.png   uint16 PNG (mm), upsampled to RGB resolution
    traj.txt                 one row-major 4×4 w2c matrix per frame
    intrinsics.txt           "fx fy cx cy width height"  (post-resize)
    frame_ids.txt            original video frame index for each kept frame

After running, update your YAML config with the intrinsics printed at the end.
"""

import argparse
import struct
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

try:
    import lz4.block
    HAS_LZ4 = True
except ImportError:
    HAS_LZ4 = False

DEPTH_H, DEPTH_W = 192, 256
DEPTH_FRAME_BYTES = DEPTH_H * DEPTH_W * 2  # uint16


# ---------------------------------------------------------------------------
# COLMAP helpers
# ---------------------------------------------------------------------------

def _quat_to_rotmat(qw, qx, qy, qz):
    return np.array([
        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
        [2*(qx*qy + qw*qz), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qw*qx)],
        [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx*qx + qy*qy)],
    ], dtype=np.float64)


def parse_colmap_cameras(cameras_txt: Path):
    """Return (fx, fy, cx, cy, k1, k2, p1, p2, width, height) from cameras.txt."""
    with open(cameras_txt) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            model = parts[1]
            w, h = int(parts[2]), int(parts[3])
            params = list(map(float, parts[4:]))
            if model == 'OPENCV':
                fx, fy, cx, cy = params[0], params[1], params[2], params[3]
                k1, k2, p1, p2 = params[4], params[5], params[6], params[7]
            elif model == 'PINHOLE':
                fx, fy, cx, cy = params[0], params[1], params[2], params[3]
                k1 = k2 = p1 = p2 = 0.0
            elif model == 'SIMPLE_PINHOLE':
                fx = fy = params[0]
                cx, cy = params[1], params[2]
                k1 = k2 = p1 = p2 = 0.0
            else:
                raise ValueError(f"Unsupported COLMAP camera model: {model}")
            return fx, fy, cx, cy, k1, k2, p1, p2, w, h
    raise RuntimeError(f"No camera entry found in {cameras_txt}")


def parse_colmap_images(images_txt: Path) -> dict:
    """Return {frame_idx: T_w2c (4×4 float64)} from images.txt.

    COLMAP images.txt format (two lines per image):
        IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
        POINTS2D[] as (X Y POINT3D_ID)   ← empty when observations aren't stored

    The rotation R and translation t give the world-to-camera transform:
        X_cam = R @ X_world + t
    which is exactly what MonoGS needs, in OpenCV camera convention.
    """
    poses: dict = {}
    with open(images_txt) as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith('#'):
            i += 1
            continue
        parts = line.split()
        if len(parts) >= 10:
            qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
            name = parts[9]  # e.g. "frame_000000.jpg"
            frame_idx = int(name.replace('frame_', '').replace('.jpg', ''))
            R = _quat_to_rotmat(qw, qx, qy, qz)
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = R
            T[:3, 3] = [tx, ty, tz]
            poses[frame_idx] = T
            i += 2  # skip the observations line
        else:
            i += 1
    return poses


# ---------------------------------------------------------------------------
# Depth reader
# ---------------------------------------------------------------------------

def iter_depth_frames(depth_path: Path):
    """Yield (frame_idx, uint16 ndarray shape (192, 256)) from depth.bin."""
    if not HAS_LZ4:
        raise RuntimeError("lz4 package not found. Install it with: pip install lz4")
    with open(depth_path, 'rb') as f:
        idx = 0
        while True:
            header = f.read(4)
            if len(header) < 4:
                return
            size = struct.unpack('<I', header)[0]
            payload = f.read(size)
            if len(payload) < size:
                raise IOError(f"Truncated depth.bin at frame {idx}")
            raw = lz4.block.decompress(payload, uncompressed_size=DEPTH_FRAME_BYTES)
            yield idx, np.frombuffer(raw, dtype=np.uint16).reshape(DEPTH_H, DEPTH_W).copy()
            idx += 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Preprocess a ScanNet++ scene for MonoGS using COLMAP poses."
    )
    p.add_argument("scene_dir", type=Path,
                   help="Raw ScanNet++ scene dir (contains rgb.mkv, depth.bin, colmap/)")
    p.add_argument("--out", type=Path, default=None,
                   help="Output directory (default: <scene>/processed)")
    p.add_argument("--scale", type=float, default=0.5,
                   help="RGB resize factor (default 0.5 → 960×720)")
    p.add_argument("--max-frames", type=int, default=None,
                   help="Optional cap on number of frames to extract")
    p.add_argument("--no-depth", action="store_true",
                   help="Skip depth extraction (useful when lz4 is not installed)")
    return p.parse_args()


def main():
    args = parse_args()
    scene = args.scene_dir.resolve()
    out = (args.out or scene / "processed").resolve()
    (out / "rgb").mkdir(parents=True, exist_ok=True)
    if not args.no_depth:
        (out / "depth").mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Intrinsics from COLMAP (bundle-adjusted, more accurate than ARKit)
    # ------------------------------------------------------------------
    fx, fy, cx, cy, k1, k2, p1, p2, src_w, src_h = parse_colmap_cameras(
        scene / "colmap" / "cameras.txt"
    )
    dst_w = int(round(src_w * args.scale))
    dst_h = int(round(src_h * args.scale))
    fx_s = fx * args.scale
    fy_s = fy * args.scale
    cx_s = cx * args.scale
    cy_s = cy * args.scale
    print(f"COLMAP intrinsics (full): fx={fx:.3f} fy={fy:.3f} cx={cx:.3f} cy={cy:.3f}")
    print(f"Scaled (×{args.scale}):   fx={fx_s:.3f} fy={fy_s:.3f} cx={cx_s:.3f} cy={cy_s:.3f}")
    print(f"Output resolution: {dst_w}×{dst_h}")

    # ------------------------------------------------------------------
    # Poses from COLMAP images.txt (w2c, OpenCV convention, no conversion)
    # ------------------------------------------------------------------
    print("\nParsing COLMAP poses from colmap/images.txt ...")
    colmap_poses = parse_colmap_images(scene / "colmap" / "images.txt")
    keep = sorted(colmap_poses.keys())
    if args.max_frames:
        keep = keep[: args.max_frames]
    keep_set = set(keep)
    print(f"  {len(keep)} frames  (indices {keep[0]} .. {keep[-1]}, "
          f"stride ≈ {keep[1] - keep[0]})")

    # ------------------------------------------------------------------
    # Write intrinsics.txt and traj.txt
    # ------------------------------------------------------------------
    with open(out / "intrinsics.txt", "w") as f:
        f.write(f"{fx_s:.6f} {fy_s:.6f} {cx_s:.6f} {cy_s:.6f} {dst_w} {dst_h}\n")

    with open(out / "traj.txt", "w") as ft, \
         open(out / "frame_ids.txt", "w") as fi:
        for idx in keep:
            T_w2c = colmap_poses[idx]
            ft.write(" ".join(f"{v:.10f}" for v in T_w2c.flatten()) + "\n")
            fi.write(f"{idx}\n")

    # ------------------------------------------------------------------
    # Extract RGB frames from rgb.mkv
    # ------------------------------------------------------------------
    print(f"\nExtracting {len(keep)} RGB frames from rgb.mkv ...")
    cap = cv2.VideoCapture(str(scene / "rgb.mkv"))
    n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n_video > 0:
        print(f"  Video reports {n_video} frames")

    frame_idx = 0
    n_extracted = 0
    pbar = tqdm(total=len(keep), desc="rgb")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx in keep_set:
            if args.scale != 1.0:
                frame = cv2.resize(frame, (dst_w, dst_h), interpolation=cv2.INTER_AREA)
            out_path = str(out / "rgb" / f"frame_{frame_idx:06d}.jpg")
            cv2.imwrite(out_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            n_extracted += 1
            pbar.update(1)
        frame_idx += 1
        if n_extracted >= len(keep):
            break
    cap.release()
    pbar.close()
    print(f"  Extracted {n_extracted} frames")

    # ------------------------------------------------------------------
    # Extract depth frames from depth.bin
    # ------------------------------------------------------------------
    if args.no_depth:
        print("\nSkipping depth extraction (--no-depth).")
    elif not HAS_LZ4:
        print("\nWARNING: lz4 not installed, skipping depth extraction.")
        print("         Install with: pip install lz4")
    else:
        print(f"\nExtracting {len(keep)} depth frames from depth.bin ...")
        max_keep = max(keep)
        n_depth = 0
        pbar = tqdm(total=len(keep), desc="depth")
        for d_idx, depth_mm in iter_depth_frames(scene / "depth.bin"):
            if d_idx in keep_set:
                depth_up = cv2.resize(depth_mm, (dst_w, dst_h),
                                      interpolation=cv2.INTER_NEAREST)
                cv2.imwrite(str(out / "depth" / f"frame_{d_idx:06d}.png"), depth_up)
                n_depth += 1
                pbar.update(1)
            if d_idx >= max_keep and n_depth >= len(keep):
                break
        pbar.close()
        print(f"  Extracted {n_depth} depth frames")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Done. Output written to: {out}")
    print(f"  RGB frames  : {n_extracted}")
    print(f"\nUpdate your YAML config (Dataset.Calibration) with:")
    print(f"  fx: {fx_s:.5f}")
    print(f"  fy: {fy_s:.5f}")
    print(f"  cx: {cx_s:.6f}")
    print(f"  cy: {cy_s:.6f}")
    print(f"  width: {dst_w}")
    print(f"  height: {dst_h}")
    print(f"  k1: {k1}")
    print(f"  k2: {k2}")
    print(f"  p1: {p1}")
    print(f"  p2: {p2}")
    print(f"  distorted: True")
    print(f"  depth_scale: 1000.0")


if __name__ == "__main__":
    main()
