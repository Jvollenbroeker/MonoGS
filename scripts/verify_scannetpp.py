"""
Verification script for ScanNet++ dataloader correctness.

Projects COLMAP sparse 3D points into video frames.  If the coordinate system
is correct the coloured dots will land precisely on the corresponding surfaces.

Also performs a depth cross-projection check: takes a processed depth map for
frame A, unprojects it to 3D, and re-projects into frame B — the reprojected
colours should match frame B's RGB.

Usage (raw data only, no preprocessing needed):
    python scripts/verify_scannetpp.py datasets/scannet/8b5caf3398

Optional: pass --processed to also run the depth cross-projection test:
    python scripts/verify_scannetpp.py datasets/scannet/8b5caf3398 \
        --processed datasets/scannet/8b5caf3398/processed

Output images are saved to --out (default: /tmp/scannetpp_verify/).
"""

import argparse
import struct
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# COLMAP parsers (duplicated here so this script is self-contained)
# ---------------------------------------------------------------------------

def _quat_to_rotmat(qw, qx, qy, qz):
    return np.array([
        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
        [2*(qx*qy + qw*qz), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qw*qx)],
        [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx*qx + qy*qy)],
    ], dtype=np.float64)


def parse_colmap_cameras(cameras_txt: Path):
    """Return K (3×3), dist_coeffs (4,), width, height."""
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
                fx, fy, cx, cy = params[:4]
                k1, k2, p1, p2 = params[4:8]
            elif model == 'PINHOLE':
                fx, fy, cx, cy = params[:4]
                k1 = k2 = p1 = p2 = 0.0
            else:
                fx = fy = params[0]
                cx, cy = params[1], params[2]
                k1 = k2 = p1 = p2 = 0.0
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
            dist = np.array([k1, k2, p1, p2], dtype=np.float64)
            return K, dist, w, h
    raise RuntimeError(f"No camera in {cameras_txt}")


def parse_colmap_images(images_txt: Path, frame_limit=None) -> dict:
    """Return {frame_idx: T_w2c (4×4)} from images.txt."""
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
            name = parts[9]
            fidx = int(name.replace('frame_', '').replace('.jpg', ''))
            R = _quat_to_rotmat(qw, qx, qy, qz)
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = R
            T[:3, 3] = [tx, ty, tz]
            poses[fidx] = T
            i += 2
        else:
            i += 1
        if frame_limit and len(poses) >= frame_limit:
            break
    return poses


def load_points3d(points_txt: Path, max_pts: int = 15000):
    """Return pts3d (N,3) and colors (N,3) uint8 from points3D.txt."""
    pts3d, colors = [], []
    with open(points_txt) as f:
        for line in f:
            if not line.strip() or line.startswith('#'):
                continue
            parts = line.split()
            pts3d.append([float(parts[1]), float(parts[2]), float(parts[3])])
            colors.append([int(parts[4]), int(parts[5]), int(parts[6])])
            if len(pts3d) >= max_pts:
                break
    return np.array(pts3d, dtype=np.float64), np.array(colors, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------

def extract_frame_from_mkv(mkv_path: Path, frame_idx: int) -> np.ndarray:
    """Extract a single frame (RGB) from an MKV by absolute frame index."""
    cap = cv2.VideoCapture(str(mkv_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, bgr = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Cannot read frame {frame_idx} from {mkv_path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def load_processed_frame(processed_dir: Path, frame_idx: int):
    """Load RGB + depth from the processed/ directory."""
    rgb_path = processed_dir / "rgb" / f"frame_{frame_idx:06d}.jpg"
    dep_path = processed_dir / "depth" / f"frame_{frame_idx:06d}.png"
    rgb = cv2.cvtColor(cv2.imread(str(rgb_path)), cv2.COLOR_BGR2RGB)
    depth = cv2.imread(str(dep_path), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0  # mm→m
    return rgb, depth


# ---------------------------------------------------------------------------
# Projection helpers
# ---------------------------------------------------------------------------

def project_points(pts3d: np.ndarray, T_w2c: np.ndarray,
                   K: np.ndarray, dist: np.ndarray,
                   img_w: int, img_h: int):
    """Project 3D world points to pixel coords.

    Returns px (N,), py (N,), valid mask (N,) for points in front + in image.
    """
    R = T_w2c[:3, :3]
    t = T_w2c[:3, 3]
    pts_cam = (R @ pts3d.T).T + t
    in_front = pts_cam[:, 2] > 0.05

    rvec, _ = cv2.Rodrigues(R)
    pts2d, _ = cv2.projectPoints(
        pts3d.astype(np.float32),
        rvec, t.reshape(3, 1).astype(np.float32),
        K.astype(np.float32),
        dist.astype(np.float32),
    )
    px = pts2d[:, 0, 0]
    py = pts2d[:, 0, 1]
    in_img = (px >= 0) & (px < img_w) & (py >= 0) & (py < img_h)
    return px, py, in_front & in_img


def draw_points(img_rgb: np.ndarray, px, py, valid, point_colors,
                radius: int = 5) -> np.ndarray:
    """Draw coloured dots (actual COLMAP colour) with a white border."""
    out = img_rgb.copy()
    idxs = np.where(valid)[0]
    for i in idxs:
        x, y = int(round(px[i])), int(round(py[i]))
        c = [int(point_colors[i, 0]), int(point_colors[i, 1]), int(point_colors[i, 2])]
        cv2.circle(out, (x, y), radius + 1, (255, 255, 255), -1)
        cv2.circle(out, (x, y), radius, c, -1)
    return out


def save_annotated(img_rgb: np.ndarray, path: Path, label: str = ""):
    """Save RGB image (BGR on disk) with optional label."""
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    if label:
        cv2.putText(bgr, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0, 0, 0), 4)
        cv2.putText(bgr, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (255, 255, 255), 2)
    cv2.imwrite(str(path), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])


# ---------------------------------------------------------------------------
# Depth cross-projection test
# ---------------------------------------------------------------------------

def depth_cross_project(rgb_A, depth_A_m, T_w2c_A, T_w2c_B, K_s, dist_s,
                         out_w, out_h):
    """Unproject depth from frame A, re-project into frame B.

    Returns a side-by-side image: frame B RGB | reprojected colours from A.
    """
    # Build undistortion map for the scaled K
    map1, map2 = cv2.initUndistortRectifyMap(
        K_s, dist_s, np.eye(3), K_s, (out_w, out_h), cv2.CV_32FC1
    )
    rgb_A_und = cv2.remap(rgb_A, map1, map2, cv2.INTER_LINEAR)

    # Pixel grid for frame A
    ys, xs = np.where(depth_A_m > 0.1)
    zs = depth_A_m[ys, xs]

    # Unproject from scaled image to 3D camera A
    fx, fy = K_s[0, 0], K_s[1, 1]
    cx, cy = K_s[0, 2], K_s[1, 2]
    pts_cam_A = np.stack([
        (xs - cx) * zs / fx,
        (ys - cy) * zs / fy,
        zs,
    ], axis=1)  # (N, 3)

    # Camera A → world → camera B
    R_A, t_A = T_w2c_A[:3, :3], T_w2c_A[:3, 3]
    R_B, t_B = T_w2c_B[:3, :3], T_w2c_B[:3, 3]
    pts_world = (R_A.T @ (pts_cam_A - t_A).T).T
    pts_cam_B = (R_B @ pts_world.T).T + t_B

    in_front = pts_cam_B[:, 2] > 0.05
    # Project with distortion
    rvec_B, _ = cv2.Rodrigues(R_B)
    pts_world_32 = pts_world.astype(np.float32)
    pts2d_B, _ = cv2.projectPoints(
        pts_world_32,
        rvec_B, t_B.reshape(3, 1).astype(np.float32),
        K_s.astype(np.float32),
        dist_s.astype(np.float32),
    )
    px_B = pts2d_B[:, 0, 0]
    py_B = pts2d_B[:, 0, 1]
    in_img = (px_B >= 0) & (px_B < out_w) & (py_B >= 0) & (py_B < out_h)
    valid = in_front & in_img

    # Collect source colours
    src_rgb = rgb_A_und[ys, xs]  # (N, 3)
    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    for i in np.where(valid)[0]:
        xi, yi = int(round(px_B[i])), int(round(py_B[i]))
        canvas[yi, xi] = src_rgb[i]
    return canvas


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Verify ScanNet++ dataloader alignment")
    p.add_argument("scene_dir", type=Path,
                   help="Raw ScanNet++ scene directory")
    p.add_argument("--processed", type=Path, default=None,
                   help="Processed directory (enables depth cross-projection test)")
    p.add_argument("--out", type=Path, default=Path("/tmp/scannetpp_verify"),
                   help="Output directory for verification images")
    p.add_argument("--n-frames", type=int, default=3,
                   help="Number of frames to verify (default 3)")
    p.add_argument("--max-pts", type=int, default=15000,
                   help="Max COLMAP sparse points to load (default 15000)")
    return p.parse_args()


def main():
    args = parse_args()
    scene = args.scene_dir.resolve()
    args.out.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("ScanNet++ dataloader verification")
    print("=" * 60)

    # ---- Load COLMAP data -----------------------------------------------
    print("\nLoading COLMAP cameras ...")
    K, dist, src_w, src_h = parse_colmap_cameras(scene / "colmap" / "cameras.txt")
    print(f"  K  = {K}")
    print(f"  dist = {dist}")
    print(f"  Image size: {src_w}×{src_h}")

    print("\nLoading COLMAP image poses ...")
    all_poses = parse_colmap_images(scene / "colmap" / "images.txt")
    frame_indices = sorted(all_poses.keys())
    print(f"  {len(frame_indices)} poses loaded")

    print(f"\nLoading up to {args.max_pts} COLMAP 3D points ...")
    pts3d, pt_colors = load_points3d(scene / "colmap" / "points3D.txt", args.max_pts)
    print(f"  Loaded {len(pts3d)} points")

    # ---- Select frames to verify ----------------------------------------
    n = args.n_frames
    step = max(1, len(frame_indices) // n)
    verify_frames = [frame_indices[i * step] for i in range(n)]
    print(f"\nVerifying frames: {verify_frames}")

    # ---- Sparse point reprojection --------------------------------------
    print("\n--- Sparse-point reprojection test (uses full-res COLMAP K) ---")
    for fidx in verify_frames:
        print(f"\n  Frame {fidx:06d}: extracting from rgb.mkv ...")
        img = extract_frame_from_mkv(scene / "rgb.mkv", fidx)

        T_w2c = all_poses[fidx]
        px, py, valid = project_points(pts3d, T_w2c, K, dist, src_w, src_h)
        n_vis = valid.sum()
        print(f"  {n_vis} / {len(pts3d)} points visible")

        annotated = draw_points(img, px, py, valid, pt_colors)
        out_path = args.out / f"sparse_reproj_frame_{fidx:06d}.jpg"
        label = f"Frame {fidx}  |  {n_vis} COLMAP pts reprojected"
        save_annotated(annotated, out_path, label)
        print(f"  Saved → {out_path}")

    # ---- Depth cross-projection test ------------------------------------
    if args.processed is not None:
        processed = args.processed.resolve()
        print("\n--- Depth cross-projection test (uses processed/ depth maps) ---")

        # Load processed intrinsics
        intr = (processed / "intrinsics.txt").read_text().split()
        fx_s, fy_s, cx_s, cy_s = float(intr[0]), float(intr[1]), float(intr[2]), float(intr[3])
        out_w, out_h = int(intr[4]), int(intr[5])
        K_s = np.array([[fx_s, 0, cx_s], [0, fy_s, cy_s], [0, 0, 1]], dtype=np.float64)
        dist_s = dist  # same lens, scaled K only

        # Load processed poses
        traj_lines = (processed / "traj.txt").read_text().splitlines()
        frame_ids_lines = (processed / "frame_ids.txt").read_text().splitlines()
        proc_poses = {}
        for lid, line in zip(frame_ids_lines, traj_lines):
            T = np.array(list(map(float, line.split())), dtype=np.float64).reshape(4, 4)
            proc_poses[int(lid)] = T

        proc_frames = sorted(proc_poses.keys())
        step2 = max(1, len(proc_frames) // (args.n_frames + 1))

        for i in range(args.n_frames):
            fidx_A = proc_frames[i * step2]
            fidx_B = proc_frames[i * step2 + step2 // 2]
            if fidx_B not in proc_poses:
                continue

            print(f"\n  A={fidx_A:06d} → B={fidx_B:06d}")
            try:
                rgb_A, dep_A = load_processed_frame(processed, fidx_A)
                rgb_B, _dep_B = load_processed_frame(processed, fidx_B)
            except Exception as e:
                print(f"  Could not load frame: {e}")
                continue

            canvas = depth_cross_project(
                rgb_A, dep_A,
                proc_poses[fidx_A], proc_poses[fidx_B],
                K_s, dist_s, out_w, out_h,
            )
            # Side-by-side: frame B | reprojected colours from A
            side = np.concatenate([rgb_B, canvas], axis=1)
            out_path = args.out / f"depth_xproj_A{fidx_A:06d}_B{fidx_B:06d}.jpg"
            save_annotated(side, out_path,
                           f"B={fidx_B} | depth from A={fidx_A} reprojected")
            print(f"  Saved → {out_path}")

    print(f"\n{'=' * 60}")
    print("PASS if coloured dots align with surfaces in sparse_reproj_*.jpg")
    print("PASS if reprojected colours match frame B in depth_xproj_*.jpg")
    print(f"Output: {args.out}")


if __name__ == "__main__":
    main()
