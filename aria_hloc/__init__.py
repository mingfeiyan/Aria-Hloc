"""Aria-Hloc: relocalize Project Aria (Gen 2) frames in an Aria MPS map with hloc.

The package is organised as:

* :mod:`aria_hloc.geometry` - numpy-only SE(3) helpers (no heavy dependencies).
* :mod:`aria_hloc.aria` - readers for MPS outputs and VRS recordings, camera
  calibration handling (fisheye -> pinhole rectification).
* :mod:`aria_hloc.mapping` - builds an hloc/COLMAP map from the MPS trajectory.
* :mod:`aria_hloc.localization` - the in-memory :class:`Relocalizer`.
* :mod:`aria_hloc.service` - FastAPI service exposing the relocalizer.
* :mod:`aria_hloc.cli` - command line entry point ``aria-hloc``.
"""

__version__ = "0.1.0"
