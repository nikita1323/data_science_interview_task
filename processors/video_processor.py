"""Abstract base class for video processing in object detection pipelines."""
from abc import ABC, abstractmethod
import cv2
import torch


class VideoProcessor(ABC):
    """Abstract base class for processing video frames in object detection.

    Args:
        input_path (str): Path to the input video file.
        output_path (str): Path where the processed video will be saved.

    Attributes:
        input_path (str): Path to the input video.
        output_path (str): Path for the output video.
        device (torch.device): Device (CUDA or CPU) for processing.
        cap (cv2.VideoCapture): Video capture object for reading frames.
        frame_width (int): Width of video frames.
        frame_height (int): Height of video frames.
        fps (int): Frames per second of the video.
        out (cv2.VideoWriter): Video writer object for saving processed frames.
    """

    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cpu":
            raise RuntimeError(
                "CUDA is not available. This script requires a GPU.")
        self.cap = cv2.VideoCapture(input_path)
        if not self.cap.isOpened():
            raise ValueError(f"Error opening video file: {input_path}")
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.out = cv2.VideoWriter(self.output_path, cv2.VideoWriter_fourcc(
            *"mp4v"), self.fps, (self.frame_width, self.frame_height))

    @abstractmethod
    def load_model(self):
        """Loads the model for inference.

        This method must be implemented by subclasses to initialize the specific
        object detection model.
        """

    @abstractmethod
    def process_batch(self, frames, result_queue):
        """Processes a batch of frames and puts results in a queue.

        Args:
            frames (list): List of frames (RGB numpy arrays) to process.
            result_queue (queue.Queue): Queue to store the detection results.

        This method must be implemented by subclasses to perform inference on the frames.
        """

    def draw_boxes(self, frame, detections, color, model_name):
        """Draws bounding boxes on a frame based on detection results.

        Args:
            frame (np.ndarray): Input frame in BGR format.
            detections (dict): Detection results with 'boxes', 'scores', and 'labels'.
            color (tuple): RGB color tuple for the bounding boxes (e.g., (0, 255, 0)).
            model_name (str): Name of the model for labeling (e.g., 'YOLO').

        Returns:
            np.ndarray: Frame with bounding boxes drawn.
        """
        frame_copy = frame.copy()
        for box, score in zip(detections["boxes"], detections["scores"]):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)
            label = f"{model_name}: {score:.2f}"
            cv2.putText(frame_copy, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return frame_copy

    def cleanup(self):
        """Releases video capture and writer resources."""
        self.cap.release()
        self.out.release()
