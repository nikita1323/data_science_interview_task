"""Initialization module for the processors package.

This module makes the processors directory a package and exposes the key classes
for use in the pipeline.
"""

from .video_processor import VideoProcessor
from .yolo_processor import YoloProcessor
from .detr_processor import DetrProcessor

__all__ = ["VideoProcessor", "YoloProcessor", "DetrProcessor"]
