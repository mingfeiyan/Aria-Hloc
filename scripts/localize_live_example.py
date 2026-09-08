"""Example client: stream frames of a VRS file to a running service and print device poses."""

import argparse

import numpy as np

from aria_hloc.aria.calibration import camera_calibration_to_dict
from aria_hloc.aria.vrs import AriaVrsReader
from aria_hloc.service.client import AriaHlocClient


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:8080")
    p.add_argument("--vrs", required=True)
    p.add_argument("--camera", default="camera-rgb")
    p.add_argument("--every", type=int, default=30)
    args = p.parse_args()

    reader = AriaVrsReader(args.vrs)
    client = AriaHlocClient(args.url)
    # Send the query device's own calibration so that a different Aria device can be localized.
    calib = camera_calibration_to_dict(reader.camera_calibration(args.camera))
    for frame in reader.iter_frames(args.camera, step=args.every):
        res = client.localize(frame.image, camera_label=args.camera, calibration=calib)
        if res["success"]:
            t = np.asarray(res["T_world_device"]["translation"])
            print(f"{frame.timestamp_ns}: inliers={res['num_inliers']:4d} t_world_device={t.round(3)}")
        else:
            print(f"{frame.timestamp_ns}: failed ({res['message']})")


if __name__ == "__main__":
    main()
