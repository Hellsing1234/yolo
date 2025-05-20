# Ultralytics YOLO 🚀, AGPL-3.0 license

__version__ = "8.2.50"

import os

# Set ENV Variables (place before imports)
os.environ["OMP_NUM_THREADS"] = "1"  # reduce CPU utilization during training

from ultralytics.data.explorer.explorer import Explorer
# from ultralytics.models.yolo import YOLO
# from ultralytics.models.nas import NAS
# from ultralytics.models.rtdetr import RTDETR
# from ultralytics.models.sam import SAM
# from ultralytics.models.fastsam import FastSAM
# from ultralytics.models.yolo import YOLOWorld
from ultralytics.models import NAS, RTDETR, SAM, YOLO, FastSAM, YOLOWorld
from ultralytics.utils import ASSETS, SETTINGS
from ultralytics.utils.checks import check_yolo as checks
from ultralytics.utils.downloads import download

settings = SETTINGS
__all__ = (
    "__version__",
    "ASSETS",
    "YOLO",
    "YOLOWorld",
    "NAS",
    "SAM",
    "FastSAM",
    "RTDETR",
    "checks",
    "download",
    "settings",
    "Explorer",
)
