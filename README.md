# Aria-Hloc

Relocalize **Project Aria (Gen 2)** frames in a map built from **Aria MPS** (Machine
Perception Services) SLAM output, using
[Hierarchical-Localization (hloc)](https://github.com/cvg/Hierarchical-Localization).

Given a VRS recording and its MPS SLAM output, `aria-hloc build-map` turns the MPS
trajectory + images into an hloc map (SuperPoint features, LightGlue matches,
COLMAP model triangulated with the MPS poses). The map lives in the MPS world frame
and is metric. `aria-hloc serve` then exposes a relocalization service: send a raw
Aria frame (any of the RGB/SLAM cameras), get back `T_world_camera` and
`T_world_device` in the MPS world frame.

```
VRS + MPS (closed_loop_trajectory.csv)          Aria frame (raw fisheye)
              │                                          │
   keyframes by motion  ──►  rectify to pinhole  ◄────── rectify to pinhole
              │                                          │
   COLMAP reference model (MPS poses)            NetVLAD retrieval
              │                                          │
   NetVLAD ▸ pairs from poses ▸ SuperPoint ▸ LightGlue    SuperPoint + LightGlue vs keyframes
              │                                          │
   triangulation with fixed poses  ──── map ────►  2D-3D matches ▸ PnP + RANSAC
                                                         │
                                                T_world_camera, T_world_device
```

## How it works

1. **Poses from MPS.** `closed_loop_trajectory.csv` gives `T_world_device` at 1 kHz.
   For every image the device pose is interpolated (SLERP) at the capture timestamp.
2. **Keyframes.** Frames are kept when the device moved ≥ `--min-translation` (0.15 m)
   or rotated ≥ `--min-rotation` (10°) since the last keyframe, per camera stream.
3. **Rectification.** Aria cameras use the `Fisheye624` model, which COLMAP/hloc do
   not support. Every image is warped to a pinhole camera with `projectaria_tools`
   (`get_linear_camera_calibration` + `distort_by_calibration`) and rotated 90° so
   it is upright (learned features prefer upright images). The pinhole camera keeps
   the optical centre and axis, so `T_world_camera = T_world_device · T_device_camera`
   with the (rotated) extrinsics from the calibration. When
   `online_calibration.jsonl` is present, the per-frame online calibration is used
   for the keyframes.
4. **hloc map.** A COLMAP *reference* model with the keyframe poses is written, then
   hloc runs: global descriptors (NetVLAD), image pairs from the known poses,
   SuperPoint, LightGlue, and `hloc.triangulation` (points triangulated with the
   poses held fixed). Result: an hloc SfM model in the MPS frame with metric scale.
5. **Relocalization.** Same algorithm as `hloc.localize_sfm`, kept in memory: retrieve
   the top-k keyframes, match the query against them, lift matches to 3D points,
   `pycolmap.estimate_and_refine_absolute_pose` (P3P + LO-RANSAC + refinement).
   `T_world_device` is recovered from the camera extrinsics, so a query from any
   camera of the rig yields the device pose.

Because the MPS semi-dense points do not carry learned descriptors, the map points
are re-triangulated from hloc features; the MPS semi-dense cloud is only used for
visual sanity checks (`export-ply`).

## Installation

```bash
# 1. hloc (must be a *recursive* clone: SuperPoint/SuperGlue are git submodules)
git clone --recursive https://github.com/cvg/Hierarchical-Localization.git
pip install -e Hierarchical-Localization          # pulls torch, pycolmap>=3.13, lightglue, ...

# 2. this package
git clone https://github.com/mingfeiyan/Aria-Hloc.git
pip install -e "Aria-Hloc[aria,service]"           # projectaria_tools, fastapi, uvicorn
```

Notes
* A CUDA GPU is strongly recommended for map building; the service also runs on CPU
  (≈2–3 s per query with the default configuration).
* On some Debian/Ubuntu Python installs `pip install hloc` fails with
  `AttributeError: install_layout`; use `pip install --no-build-isolation -e Hierarchical-Localization`
  or add the clone to `PYTHONPATH`.
* NetVLAD weights are downloaded on first use from `cvg-data.inf.ethz.ch`. Behind a
  restrictive proxy you can pick another hloc retrieval conf, e.g.
  `--retrieval openibl` (weights from GitHub releases).
* Aria Gen 2 recordings need `projectaria_tools >= 2.0`.
* A `Dockerfile` (CUDA runtime) is provided: `docker build -t aria-hloc .`

## Data preparation

1. Record with Aria Gen 2 → `recording.vrs`.
2. Request **MPS SLAM** for the recording (Aria Studio / Desktop app / MPS CLI). You
   get a folder like:
   ```
   mps_recording_vrs/
     slam/
       closed_loop_trajectory.csv     ← required
       online_calibration.jsonl       ← optional, used when present
       semidense_points.csv.gz        ← optional (export-ply only)
       semidense_observations.csv.gz
   ```
3. Keep the VRS: MPS output contains no images, they are read from the VRS.

To evaluate queries with ground truth, process the query recording with MPS too. If
both recordings were processed together with **multi-sequence SLAM** they share the
world frame; otherwise pass `--align` to the evaluation (rigid alignment of the
estimated trajectory to the query's own MPS frame).

## Usage

### Build a map

```bash
aria-hloc build-map \
    --vrs recording.vrs \
    --mps mps_recording_vrs \
    --output maps/office \
    --cameras camera-rgb camera-slam-front-left camera-slam-front-right   # default: all RGB/SLAM streams
```

Useful options: `--min-translation/--min-rotation` (keyframe density),
`--max-keyframes N` (cap per camera), `--rectified-scale 0.5` (smaller images, faster),
`--focal-scale 0.8` (wider pinhole field of view), `--pairs-from both`
(also use retrieval pairs, for revisits that the pose graph misses),
`--features/--matcher/--retrieval` (any hloc conf name), `--keyframes-only`.

`aria-hloc info --map maps/office` prints the metadata and statistics;
`aria-hloc export-ply --map maps/office --output office.ply --mps mps_recording_vrs`
writes the triangulated points (colour), keyframe positions (red) and the MPS
semi-dense points (blue) to one PLY for a visual check that everything is in the
same frame.

Map layout (`aria_hloc/map_format.py`):

```
maps/office/
  map.json               cameras (pinhole + T_device_camera + source fisheye calibration), confs, stats
  keyframes.json         name, label, timestamp, T_world_camera, T_world_device per keyframe
  images/<label>/<ts>.jpg
  reference/             COLMAP model with MPS poses
  sfm/                   triangulated COLMAP model (used for localization)
  features.h5, global-features.h5, pairs.txt, matches.h5
```

### Localize frames of a recording (batch + evaluation)

```bash
aria-hloc localize --map maps/office --vrs query.vrs --every 30 \
    --mps mps_query_vrs --align --output results.json
```

Prints recall at (5 cm, 2°), (25 cm, 5°), (1 m, 10°) and median errors versus the
query's MPS trajectory; `results.json` contains every frame with its estimated
`T_world_device`, inlier counts, timings and errors. `aria-hloc evaluate --results
results.json --mps ...` re-evaluates a results file. `--images DIR --camera-label
camera-rgb` localizes a folder of raw frames (or `--rectified` pinhole images).

### Run the service

```bash
aria-hloc serve --map maps/office --port 8080            # --device cuda|cpu, --num-retrieved 10
```

| Endpoint | Description |
| --- | --- |
| `GET /health` | liveness, whether the map is loaded |
| `GET /map` | map metadata: cameras, statistics, configuration |
| `POST /localize` | multipart: `image` (PNG/JPEG) + form fields `camera_label`, `rectified`, `calibration`, `num_retrieved` |
| `POST /localize/json` | same with `image_base64` in a JSON body |

* Raw Aria frames (`rectified=false`, default): `camera_label` selects the camera
  (`camera-rgb`, `camera-slam-front-left`, ...). The service rectifies the frame with
  the calibration stored in the map, or with the calibration of the *query* device if
  `calibration` carries it (`aria_hloc.aria.calibration.camera_calibration_to_dict`).
* Pinhole images (`rectified=true`): `calibration` = `{"width","height","fx","fy","cx","cy"}`.

Response:

```json
{
  "success": true,
  "T_world_camera": {"matrix": [[...]], "quaternion_wxyz": [...], "translation": [...]},
  "T_world_device": {"matrix": [[...]], "quaternion_wxyz": [...], "translation": [...]},
  "num_inliers": 412, "num_matches": 655, "num_keypoints": 2048, "inlier_ratio": 0.63,
  "retrieved": ["camera-rgb/1234.jpg", "..."],
  "timings_ms": {"rectify": 12.0, "retrieval": 15.1, "features": 9.8, "matching": 60.2, "pnp": 4.0},
  "message": ""
}
```

Python client:

```python
from aria_hloc.aria.vrs import AriaVrsReader
from aria_hloc.aria.calibration import camera_calibration_to_dict
from aria_hloc.service.client import AriaHlocClient

reader = AriaVrsReader("query.vrs")
client = AriaHlocClient("http://localhost:8080")
calib = camera_calibration_to_dict(reader.camera_calibration("camera-rgb"))
for frame in reader.iter_frames("camera-rgb", step=30):
    res = client.localize(frame.image, camera_label="camera-rgb", calibration=calib)
    if res["success"]:
        print(frame.timestamp_ns, res["T_world_device"]["translation"], res["num_inliers"])
```

See `scripts/localize_live_example.py`. In-process use without HTTP:

```python
from aria_hloc.localization import Relocalizer
reloc = Relocalizer("maps/office")
res = reloc.localize_aria_frame(frame.image, "camera-rgb", src_calib=reader.camera_calibration("camera-rgb"))
```

## Conventions

* `T_a_b` is a 4×4 matrix mapping points from frame `b` to frame `a`; camera frames
  are x-right, y-down, z-forward (Aria and COLMAP agree). Quaternions in JSON are
  `[w, x, y, z]`.
* `T_world_device` refers to the Aria *device* frame used by MPS, so results can be
  compared with `closed_loop_trajectory.csv` directly and combined with other MPS
  outputs (eye gaze, hand tracking, semi-dense points).
* Timestamps are device-time nanoseconds (`capture_timestamp_ns`). If the calibration
  provides `time_offset_sec_device_camera` it is added before the pose lookup
  (disable with `--no-time-offset`).

## Tests

```bash
pip install -e ".[dev]"
pytest                          # unit tests: geometry, MPS parsing, keyframes, COLMAP model, service (no torch needed)
ARIA_HLOC_E2E=1 pytest tests/test_e2e_synthetic.py -s   # full hloc map build + relocalization on a synthetic scene
```

The end-to-end test renders a textured plane from known poses, builds a map with hloc,
and relocalizes held-out views (expected: < 5 cm / 1°, typically millimetres).

## Limitations / next steps

* Map building uses the factory calibration (or MPS online calibration when present);
  it does not refine poses (the MPS trajectory is trusted).
* Queries are localized one image at a time; a multi-camera query (all rig cameras at
  once with a rig PnP) and temporal filtering are natural extensions.
* Retrieval is global-descriptor based; a pose prior (e.g. from the on-device VIO) is
  not yet used to restrict the search.
