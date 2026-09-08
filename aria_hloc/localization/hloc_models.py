"""In-memory wrappers around hloc's feature extractors, matchers and retrieval nets.

hloc's ``extract_features``/``match_features`` work on image folders and HDF5
files. A service needs to process a single image in memory, so these classes
reproduce hloc's preprocessing (resize, grayscale, keypoint rescaling) and run
the same models on numpy arrays. The outputs follow the hloc HDF5 layout
(``keypoints`` (N, 2), ``descriptors`` (D, N), ``scores`` (N,), ``image_size``
(2,)) so that map features written by hloc and query features computed here are
interchangeable.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


def get_device(device: Optional[str] = None) -> str:
    import torch

    if device is None or device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _preprocess(image: np.ndarray, conf: Dict) -> Dict:
    """Mirror ``hloc.extract_features.ImageDataset.__getitem__`` for a numpy image (RGB or gray)."""
    import cv2
    import torch
    from hloc.extract_features import resize_image

    grayscale = bool(conf.get("grayscale", False))
    resize_max = conf.get("resize_max")
    resize_force = bool(conf.get("resize_force", False))
    interpolation = conf.get("interpolation", "cv2_area")

    if image.ndim == 2:
        if not grayscale:
            image = np.repeat(image[:, :, None], 3, axis=2)
    elif grayscale:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = image.astype(np.float32)
    size = image.shape[:2][::-1]
    if resize_max and (resize_force or max(size) > resize_max):
        scale = resize_max / max(size)
        size_new = tuple(int(round(x * scale)) for x in size)
        image = resize_image(image, size_new, interpolation)
    if grayscale:
        image = image[None]
    else:
        image = image.transpose((2, 0, 1))
    image = image / 255.0
    return {"image": torch.from_numpy(np.ascontiguousarray(image))[None], "original_size": np.array(size)}


class LocalFeatureExtractor:
    """Run an hloc local feature extractor (e.g. SuperPoint) on one image."""

    def __init__(self, conf_name: str = "superpoint_aachen", device: Optional[str] = None):
        import torch
        from hloc import extract_features, extractors
        from hloc.utils.base_model import dynamic_load

        self.conf = extract_features.confs[conf_name]
        self.device = get_device(device)
        Model = dynamic_load(extractors, self.conf["model"]["name"])
        self.model = Model(self.conf["model"]).eval().to(self.device)
        self._torch = torch

    def __call__(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        data = _preprocess(image, self.conf["preprocessing"])
        with self._torch.no_grad():
            pred = self.model({"image": data["image"].to(self.device, non_blocking=True)})
        pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
        original_size = data["original_size"]
        pred["image_size"] = original_size
        if "keypoints" in pred:
            size = np.array(data["image"].shape[-2:][::-1])
            scales = (original_size / size).astype(np.float32)
            pred["keypoints"] = (pred["keypoints"] + 0.5) * scales[None] - 0.5
            if "scales" in pred:
                pred["scales"] *= scales.mean()
            pred["uncertainty"] = float(getattr(self.model, "detection_noise", 1) * scales.mean())
        return pred


class GlobalDescriptorExtractor:
    """Run an hloc global descriptor network (e.g. NetVLAD) on one image."""

    def __init__(self, conf_name: str = "netvlad", device: Optional[str] = None):
        import torch
        from hloc import extract_features, extractors
        from hloc.utils.base_model import dynamic_load

        self.conf = extract_features.confs[conf_name]
        self.device = get_device(device)
        Model = dynamic_load(extractors, self.conf["model"]["name"])
        self.model = Model(self.conf["model"]).eval().to(self.device)
        self._torch = torch

    def __call__(self, image: np.ndarray) -> np.ndarray:
        data = _preprocess(image, self.conf["preprocessing"])
        with self._torch.no_grad():
            pred = self.model({"image": data["image"].to(self.device, non_blocking=True)})
        return pred["global_descriptor"][0].cpu().numpy().astype(np.float32)


class FeatureMatcher:
    """Run an hloc matcher (e.g. LightGlue) on two feature sets."""

    def __init__(self, conf_name: str = "superpoint+lightglue", device: Optional[str] = None):
        import torch
        from hloc import match_features, matchers
        from hloc.utils.base_model import dynamic_load

        self.conf = match_features.confs[conf_name]
        self.device = get_device(device)
        Model = dynamic_load(matchers, self.conf["model"]["name"])
        self.model = Model(self.conf["model"]).eval().to(self.device)
        self._torch = torch

    def _to_tensor(self, feats: Dict[str, np.ndarray], suffix: str) -> Dict:
        torch = self._torch
        data = {}
        for k, v in feats.items():
            if k in ("image_size", "uncertainty"):
                continue
            data[k + suffix] = torch.from_numpy(np.asarray(v, dtype=np.float32))[None].to(self.device)
        w, h = [int(x) for x in np.asarray(feats["image_size"]).reshape(2)]
        data["image" + suffix] = torch.empty((1, 1, h, w))
        return data

    def __call__(self, feats0: Dict[str, np.ndarray], feats1: Dict[str, np.ndarray]):
        """Return ``(matches (M, 2) int, scores (M,))`` with indices into feats0/feats1."""
        data = {**self._to_tensor(feats0, "0"), **self._to_tensor(feats1, "1")}
        with self._torch.no_grad():
            pred = self.model(data)
        matches0 = pred["matches0"][0].cpu().numpy()
        idx = np.where(matches0 != -1)[0]
        matches = np.stack([idx, matches0[idx]], -1).astype(np.int64) if len(idx) else np.zeros((0, 2), np.int64)
        if "matching_scores0" in pred:
            scores = pred["matching_scores0"][0].cpu().numpy()[idx]
        else:
            scores = np.ones(len(idx), dtype=np.float32)
        return matches, scores
