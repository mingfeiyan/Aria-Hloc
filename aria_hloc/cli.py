"""Command line interface: ``aria-hloc <command>``.

Commands
--------
build-map   Build a map from a VRS recording + its MPS SLAM output.
localize    Localize frames of a query VRS (or an image folder) in a map,
            optionally evaluating against the query's MPS trajectory.
evaluate    Evaluate a results file against an MPS trajectory.
serve       Start the HTTP relocalization service.
info        Print map information.
export-ply  Export the map 3D points and keyframe positions as PLY.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from . import __version__
from .geometry import transform_to_dict

logger = logging.getLogger("aria_hloc")


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="[%(asctime)s %(name)s %(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def _add_reloc_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--num-retrieved", type=int, default=10, help="keyframes retrieved per query")
    p.add_argument("--ransac-thresh", type=float, default=12.0, help="PnP RANSAC reprojection threshold (px)")
    p.add_argument("--min-inliers", type=int, default=12)
    p.add_argument("--covisibility-clustering", action="store_true")
    p.add_argument("--same-camera-only", action="store_true", help="retrieve only keyframes of the query camera")
    p.add_argument("--device", default=None, help="torch device (default: cuda if available)")


def _reloc_config(args):
    from .localization.relocalizer import RelocalizerConfig

    return RelocalizerConfig(
        num_retrieved=args.num_retrieved,
        ransac_max_error_px=args.ransac_thresh,
        min_num_inliers=args.min_inliers,
        covisibility_clustering=args.covisibility_clustering,
        same_camera_only=args.same_camera_only,
        device=args.device,
    )


# ----------------------------------------------------------------- build-map
def cmd_build_map(args) -> int:
    from .mapping.build_map import MapBuildConfig, build_map

    cfg = MapBuildConfig(
        vrs=Path(args.vrs),
        mps=Path(args.mps),
        output=Path(args.output),
        cameras=args.cameras,
        min_translation_m=args.min_translation,
        min_rotation_deg=args.min_rotation,
        max_keyframes_per_camera=args.max_keyframes,
        max_pose_gap_ms=args.max_pose_gap_ms,
        min_quality=args.min_quality,
        rectified_scale=args.rectified_scale,
        focal_scale=args.focal_scale,
        upright=not args.no_upright,
        use_online_calibration=not args.no_online_calibration,
        apply_time_offset=not args.no_time_offset,
        image_format=args.image_format,
        feature_conf=args.features,
        matcher_conf=args.matcher,
        retrieval_conf=args.retrieval,
        pairs_from=args.pairs_from,
        num_pairs=args.num_pairs,
        skip_geometric_verification=args.skip_geometric_verification,
        verbose=args.verbose,
        overwrite=args.overwrite,
        keyframes_only=args.keyframes_only,
    )
    build_map(cfg)
    return 0


# ------------------------------------------------------------------ localize
def _frame_record(label: str, timestamp_ns: int, index: int, result) -> Dict[str, Any]:
    d = result.to_dict()
    d.update({"label": label, "timestamp_ns": int(timestamp_ns), "index": int(index)})
    return d


def cmd_localize(args) -> int:
    from .localization.relocalizer import Relocalizer

    reloc = Relocalizer(Path(args.map), _reloc_config(args))
    frames: List[Dict[str, Any]] = []
    t_start = time.time()

    if args.vrs:
        from .aria.vrs import AriaVrsReader

        reader = AriaVrsReader(Path(args.vrs))
        labels = reader.select_labels(args.cameras) if args.cameras else [l for l in reader.camera_labels() if l in reloc.cameras]
        labels = [l for l in labels if l in reloc.cameras]
        if not labels:
            logger.error("None of the query cameras %s are in the map %s", reader.camera_labels(), list(reloc.cameras))
            return 1
        for label in labels:
            n = reader.num_frames(label)
            indices = list(range(args.start, n, args.every))
            if args.max_frames:
                indices = indices[: args.max_frames]
            calib = reader.camera_calibration(label)
            logger.info("Localizing %d frames of %s", len(indices), label)
            for k, idx in enumerate(indices):
                frame = reader.get_frame(label, idx)
                res = reloc.localize_aria_frame(frame.image, label, calib, args.num_retrieved)
                frames.append(_frame_record(label, frame.timestamp_ns, idx, res))
                if (k + 1) % 20 == 0:
                    ok = sum(f["success"] for f in frames)
                    logger.info("  %d/%d done, %d localized", k + 1, len(indices), ok)
    elif args.images:
        import cv2

        from .aria.calibration import PinholeCamera

        root = Path(args.images)
        paths = sorted(p for p in root.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        if not paths:
            logger.error("No images found in %s", root)
            return 1
        if not args.camera_label:
            logger.error("--camera-label is required with --images")
            return 1
        pinhole = None
        if args.rectified:
            pinhole = PinholeCamera.from_dict(json.loads(args.intrinsics)) if args.intrinsics else reloc.cameras[args.camera_label].pinhole
        for k, p in enumerate(paths):
            img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            if img is None:
                logger.warning("Could not read %s", p)
                continue
            if img.ndim == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if args.rectified:
                res = reloc.localize_image(img, pinhole, args.camera_label, args.num_retrieved)
            else:
                res = reloc.localize_aria_frame(img, args.camera_label, None, args.num_retrieved)
            ts = int(p.stem) if p.stem.isdigit() else -1
            rec = _frame_record(args.camera_label, ts, k, res)
            rec["image"] = str(p.relative_to(root))
            frames.append(rec)
    else:
        logger.error("Provide --vrs or --images")
        return 1

    num_ok = sum(f["success"] for f in frames)
    logger.info("Localized %d / %d frames in %.1f s", num_ok, len(frames), time.time() - t_start)
    output: Dict[str, Any] = {"map": str(args.map), "query": args.vrs or args.images, "frames": frames}

    if args.mps:
        from .aria.mps import find_mps_paths, read_trajectory_csv
        from .evaluate import attach_errors, evaluate_frames

        traj = read_trajectory_csv(find_mps_paths(Path(args.mps)).closed_loop_trajectory)
        summary = evaluate_frames(frames, traj, align=args.align)
        T_align = None if summary.alignment is None else np.asarray(summary.alignment["T_gt_from_map"])
        attach_errors(frames, traj, T_align)
        output["evaluation"] = summary.to_dict()
        print(json.dumps(summary.to_dict(), indent=2))

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(output, f, indent=1)
        logger.info("Results written to %s", args.output)
    return 0


def cmd_evaluate(args) -> int:
    from .aria.mps import find_mps_paths, read_trajectory_csv
    from .evaluate import evaluate_frames

    with open(args.results) as f:
        results = json.load(f)
    traj = read_trajectory_csv(find_mps_paths(Path(args.mps)).closed_loop_trajectory)
    summary = evaluate_frames(results["frames"], traj, align=args.align)
    print(json.dumps(summary.to_dict(), indent=2))
    return 0


def cmd_serve(args) -> int:
    from .service.app import serve

    serve(Path(args.map), host=args.host, port=args.port, config=_reloc_config(args))
    return 0


def cmd_info(args) -> int:
    from .map_format import MapMetadata, MapPaths, load_keyframes

    paths = MapPaths(Path(args.map))
    meta = MapMetadata.load(paths.root)
    d = meta.to_dict()
    d["complete"] = paths.is_complete()
    d["num_keyframes"] = len(load_keyframes(paths.root))
    if paths.is_complete():
        import pycolmap

        rec = pycolmap.Reconstruction(str(paths.sfm))
        d["sfm"] = {"num_images": len(rec.images), "num_points3D": len(rec.points3D)}
    print(json.dumps(d, indent=2))
    return 0


def _write_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    with open(path, "wb") as f:
        header = (
            "ply\nformat binary_little_endian 1.0\n"
            f"element vertex {len(xyz)}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
        )
        f.write(header.encode("ascii"))
        dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
        arr = np.empty(len(xyz), dtype=dt)
        arr["x"], arr["y"], arr["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        arr["r"], arr["g"], arr["b"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
        f.write(arr.tobytes())


def cmd_export_ply(args) -> int:
    import pycolmap

    from .map_format import MapPaths, load_keyframes

    paths = MapPaths(Path(args.map))
    xyz_all, rgb_all = [], []
    if paths.is_complete():
        rec = pycolmap.Reconstruction(str(paths.sfm))
        if len(rec.points3D):
            xyz_all.append(np.array([p.xyz for p in rec.points3D.values()]))
            rgb_all.append(np.array([p.color for p in rec.points3D.values()], dtype=np.uint8))
    kfs = load_keyframes(paths.root)
    if kfs:
        xyz_all.append(np.array([k.T_world_camera[:3, 3] for k in kfs]))
        rgb_all.append(np.tile(np.array([[255, 0, 0]], dtype=np.uint8), (len(kfs), 1)))
    if args.mps:
        from .aria.mps import filter_semidense_points, find_mps_paths, read_semidense_points

        mp = find_mps_paths(Path(args.mps))
        if mp.semidense_points is not None:
            pts = filter_semidense_points(read_semidense_points(mp.semidense_points))
            xyz_all.append(pts.xyz)
            rgb_all.append(np.tile(np.array([[0, 160, 255]], dtype=np.uint8), (len(pts.xyz), 1)))
    if not xyz_all:
        logger.error("Nothing to export")
        return 1
    _write_ply(Path(args.output), np.concatenate(xyz_all), np.concatenate(rgb_all))
    logger.info("Wrote %s", args.output)
    return 0


# --------------------------------------------------------------------- main
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="aria-hloc", description="Relocalize Aria frames in an MPS map with hloc")
    p.add_argument("--version", action="version", version=f"aria-hloc {__version__}")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    b = sub.add_parser("build-map", help="build a map from a VRS + MPS SLAM output")
    b.add_argument("--vrs", required=True, help="Aria VRS recording")
    b.add_argument("--mps", required=True, help="MPS output folder (or closed_loop_trajectory.csv)")
    b.add_argument("--output", required=True, help="map directory to create")
    b.add_argument("--cameras", nargs="+", default=None, help="camera labels (default: all RGB/SLAM streams)")
    b.add_argument("--min-translation", type=float, default=0.15, help="keyframe spacing (m)")
    b.add_argument("--min-rotation", type=float, default=10.0, help="keyframe spacing (deg)")
    b.add_argument("--max-keyframes", type=int, default=None, help="cap per camera")
    b.add_argument("--max-pose-gap-ms", type=float, default=100.0)
    b.add_argument("--min-quality", type=float, default=None, help="minimum MPS quality_score")
    b.add_argument("--rectified-scale", type=float, default=1.0, help="rectified image size relative to source")
    b.add_argument("--focal-scale", type=float, default=1.0, help="rectified focal relative to source")
    b.add_argument("--no-upright", action="store_true", help="keep the native (rotated) Aria orientation")
    b.add_argument("--no-online-calibration", action="store_true", help="ignore MPS online_calibration.jsonl")
    b.add_argument("--no-time-offset", action="store_true", help="do not apply the camera time offset before pose lookup")
    b.add_argument("--image-format", default="jpg", choices=["jpg", "png"])
    b.add_argument("--features", default="superpoint_aachen", help="hloc local feature conf")
    b.add_argument("--matcher", default="superpoint+lightglue", help="hloc matcher conf")
    b.add_argument("--retrieval", default="netvlad", help="hloc global descriptor conf")
    b.add_argument("--pairs-from", default="poses", choices=["poses", "retrieval", "both"])
    b.add_argument("--num-pairs", type=int, default=20)
    b.add_argument("--skip-geometric-verification", action="store_true")
    b.add_argument("--overwrite", action="store_true")
    b.add_argument("--keyframes-only", action="store_true", help="stop after keyframe extraction")
    b.set_defaults(func=cmd_build_map)

    l = sub.add_parser("localize", help="localize query frames in a map")
    l.add_argument("--map", required=True)
    src = l.add_mutually_exclusive_group(required=True)
    src.add_argument("--vrs", help="query VRS recording")
    src.add_argument("--images", help="folder of query images")
    l.add_argument("--cameras", nargs="+", default=None, help="VRS camera labels to localize")
    l.add_argument("--camera-label", default=None, help="camera label of the images in --images")
    l.add_argument("--rectified", action="store_true", help="--images are already pinhole images")
    l.add_argument("--intrinsics", default=None, help='pinhole intrinsics JSON for --rectified images')
    l.add_argument("--every", type=int, default=10, help="localize every N-th frame")
    l.add_argument("--start", type=int, default=0)
    l.add_argument("--max-frames", type=int, default=None)
    l.add_argument("--mps", default=None, help="query MPS output for evaluation")
    l.add_argument("--align", action="store_true", help="rigidly align to the query MPS frame before evaluating")
    l.add_argument("--output", default=None, help="results JSON")
    _add_reloc_args(l)
    l.set_defaults(func=cmd_localize)

    e = sub.add_parser("evaluate", help="evaluate a results JSON against an MPS trajectory")
    e.add_argument("--results", required=True)
    e.add_argument("--mps", required=True)
    e.add_argument("--align", action="store_true")
    e.set_defaults(func=cmd_evaluate)

    s = sub.add_parser("serve", help="run the HTTP service")
    s.add_argument("--map", required=True)
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8080)
    _add_reloc_args(s)
    s.set_defaults(func=cmd_serve)

    i = sub.add_parser("info", help="print map information")
    i.add_argument("--map", required=True)
    i.set_defaults(func=cmd_info)

    x = sub.add_parser("export-ply", help="export map points/keyframes (and MPS semi-dense points) as PLY")
    x.add_argument("--map", required=True)
    x.add_argument("--output", required=True)
    x.add_argument("--mps", default=None)
    x.set_defaults(func=cmd_export_ply)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    _setup_logging(args.verbose)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
