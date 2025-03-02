"""YOLO-specific video processor for object detection."""

from ultralytics import YOLO
import torch
from .video_processor import VideoProcessor


class YoloProcessor(VideoProcessor):
    """Processor for performing person detection using the YOLO model.

    Args:
        input_path (str): Path to the input video file.
        output_path (str): Path where the processed video will be saved.
        model_name (str, optional): Name or path to the YOLO model file. Defaults to 'yolo11n.pt'.
        batch_size (int): Number of frames to process in each batch.
        conf_threshold (float, optional): Confidence threshold for detections. Defaults to 0.5.
        iou_threshold (float, optional): IoU threshold for NMS. Defaults to 0.7.
        if_half (bool, optional): Use FP16 (half-precision) inference. Defaults to False.
        if_verbose (bool, optional): Enable verbose output during inference. Defaults to True.

    Attributes:
        model (ultralytics.YOLO): Loaded YOLO model instance.
    """

    def __init__(self, input_path, output_path, model_name="yolo12n.pt", batch_size=8,
                 conf_threshold=0.25, iou_threshold=0.85, if_half=False, if_verbose=False):
        super().__init__(input_path, output_path)
        self.model_name = model_name
        self.batch_size = batch_size
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.if_half = if_half
        self.if_verbose = if_verbose

    def load_model(self):
        """Loads the YOLO model with specified precision settings."""
        self.model = YOLO(self.model_name).to(self.device)
        if self.if_half:
            self.model.half()

    def process_batch(self, frames, result_queue):
        """Processes a batch of frames using YOLO and stores results in a queue.

        Args:
            frames (list): List of frames (RGB numpy arrays) to process.
            result_queue (queue.Queue): Queue to store detection results as a list of dicts.
        """
        with torch.no_grad():
            results = self.model(
                frames,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                device="cuda",
                half=self.if_half,
                verbose=self.if_verbose
            )

        detections = []
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            labels = result.boxes.cls.cpu().numpy()
            person_mask = labels == 0  # "person" class in COCO
            detections.append({
                "boxes": boxes[person_mask],
                "scores": scores[person_mask],
                "labels": labels[person_mask]
            })
        result_queue.put(detections)
