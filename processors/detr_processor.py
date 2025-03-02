"""DETR-specific video processor for object detection."""

from transformers import DetrImageProcessor, DetrForObjectDetection, logging
import torch
import torchvision.ops
from .video_processor import VideoProcessor

# Suppress DETR loading warnings
logging.set_verbosity_error()


class DetrProcessor(VideoProcessor):
    """Processor for performing person detection using the DETR model.

    Args:
        input_path (str): Path to the input video file.
        output_path (str): Path where the processed video will be saved.
        model_name (str, optional): Name of the DETR model from Hugging Face. Defaults to 'facebook/detr-resnet-50'.
        batch_size (int): Number of frames to process in each batch.
        conf_threshold (float, optional): Confidence threshold for detections. Defaults to 0.5.
        iou_threshold (float, optional): IoU threshold for NMS. Defaults to 0.5.
        if_verbose (bool, optional): Enable verbose output during inference. Defaults to False.

    Attributes:
        processor (DetrImageProcessor): Preprocessor for DETR inputs.
        model (DetrForObjectDetection): Loaded DETR model instance.
    """

    def __init__(self, input_path, output_path, model_name="facebook/detr-resnet-50",
                 batch_size=8, conf_threshold=0.5, iou_threshold=0.1, if_verbose=False):
        super().__init__(input_path, output_path)
        self.model_name = model_name
        self.batch_size = batch_size
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.if_verbose = if_verbose

    def load_model(self):
        """Loads the DETR model and its preprocessor from the specified model name."""
        self.processor = DetrImageProcessor.from_pretrained(self.model_name)
        self.model = DetrForObjectDetection.from_pretrained(
            self.model_name).to(self.device)

    def process_batch(self, frames, result_queue):
        """Processes a batch of frames using DETR and stores results in a queue.

        Args:
            frames (list): List of frames (RGB numpy arrays) to process.
            result_queue (queue.Queue): Queue to store detection results as a list of dicts.

        Notes:
            If if_verbose is True, prints details such as the number of persons detected per batch.
        """
        inputs = self.processor(
            images=frames, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = torch.tensor([self.frame_height, self.frame_width]).unsqueeze(
            0).repeat(len(frames), 1).to(self.device)
        results = self.processor.post_process_object_detection(
            outputs,
            target_sizes=target_sizes,
            threshold=self.conf_threshold
        )[0]

        if len(results["boxes"]) > 0:
            boxes = results["boxes"].cpu()
            scores = results["scores"].cpu()
            keep = torchvision.ops.nms(boxes, scores, self.iou_threshold)
            results = {k: v[keep] for k, v in results.items()}

        person_mask = results["labels"] == 1  # "person" class in DETR's COCO
        detections = {
            "boxes": results["boxes"][person_mask].cpu().numpy(),
            "scores": results["scores"][person_mask].cpu().numpy(),
            "labels": results["labels"][person_mask].cpu().numpy()
        }

        if self.if_verbose:
            num_persons = len(detections["boxes"])
            print(
                f"DETR: Detected {num_persons} persons in batch of {len(frames)} frames")

        # Duplicate for consistency
        result_queue.put([detections] * len(frames))
