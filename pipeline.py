"""Pipeline module for orchestrating YOLO and DETR video processing."""

import queue
import threading
from tqdm import tqdm
import cv2
from processors import YoloProcessor, DetrProcessor


class Pipeline:
    """Orchestrates video processing with YOLO and/or DETR models.

    Args:
        args (argparse.Namespace): Command-line arguments containing configuration options.

    Attributes:
        input_path (str): Path to the input video.
        mode (str): Inference mode ('yolo_only', 'detr_only', or 'all').
        batch_size (int): Number of frames to process in each batch.
        yolo_processor (YoloProcessor): YOLO processor instance, if active.
        detr_processor (DetrProcessor): DETR processor instance, if active.
        yolo_queue (queue.Queue): Queue for YOLO detection results, if active.
        detr_queue (queue.Queue): Queue for DETR detection results, if active.

    Raises:
        ValueError: If an invalid mode is specified.

    Notes:
        Processes video in batches with a unified batch_size for all models.
    """

    def __init__(self, args):
        self.input_path = args.input_path
        self.mode = args.mode.lower()
        self.batch_size = args.batch_size
        self.yolo_processor = None
        self.detr_processor = None
        self.yolo_queue = queue.Queue() if self.mode in [
            "yolo_only", "all"] else None
        self.detr_queue = queue.Queue() if self.mode in [
            "detr_only", "all"] else None

        if self.mode not in ["yolo_only", "detr_only", "all"]:
            raise ValueError(
                "Mode must be 'yolo_only', 'detr_only', or 'all'")

        if self.mode in ["yolo_only", "all"]:
            self.yolo_processor = YoloProcessor(
                self.input_path,
                args.yolo_output_path,
                model_name=args.yolo_model_name,
                batch_size=self.batch_size,
                conf_threshold=args.yolo_conf_threshold,
                iou_threshold=args.yolo_iou_threshold,
                if_half=args.yolo_if_half,
                if_verbose=args.yolo_if_verbose
            )
            self.yolo_processor.load_model()

        if self.mode in ["detr_only", "all"]:
            self.detr_processor = DetrProcessor(
                self.input_path,
                args.detr_output_path,
                model_name=args.detr_model_name,
                batch_size=self.batch_size,
                conf_threshold=args.detr_conf_threshold,
                iou_threshold=args.detr_iou_threshold,
                if_verbose=args.detr_if_verbose
            )
            self.detr_processor.load_model()

        self.cap = (
            self.yolo_processor.cap if self.yolo_processor else
            self.detr_processor.cap
        )
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def run(self):
        """Runs the video processing pipeline in batches based on the specified mode."""
        frames_batch = []
        processed_frames = 0

        with tqdm(total=self.total_frames, desc="Processing Video", unit="frame") as pbar:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret and not frames_batch:
                    break
                if ret:
                    frames_batch.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    processed_frames += 1

                if len(frames_batch) == self.batch_size or (not ret and frames_batch):
                    threads = []
                    if self.mode in ["yolo_only", "all"]:
                        yolo_thread = threading.Thread(target=self.yolo_processor.process_batch,
                                                       args=(frames_batch, self.yolo_queue))
                        threads.append(yolo_thread)
                        yolo_thread.start()
                    if self.mode in ["detr_only", "all"]:
                        detr_thread = threading.Thread(target=self.detr_processor.process_batch,
                                                       args=(frames_batch, self.detr_queue))
                        threads.append(detr_thread)
                        detr_thread.start()

                    for thread in threads:
                        thread.join()

                    yolo_detections = self.yolo_queue.get() if self.mode in [
                        "yolo_only", "all"] else None
                    detr_detections = self.detr_queue.get() if self.mode in [
                        "detr_only", "all"] else None

                    for i, frame in enumerate(frames_batch):
                        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        if self.mode in ["yolo_only", "all"] and yolo_detections:
                            yolo_frame = self.yolo_processor.draw_boxes(frame_bgr, yolo_detections[i],
                                                                        color=(0, 255, 0), model_name="YOLO")
                            self.yolo_processor.out.write(yolo_frame)
                        if self.mode in ["detr_only", "all"] and detr_detections:
                            detr_frame = self.detr_processor.draw_boxes(frame_bgr, detr_detections[i],
                                                                        color=(255, 0, 0), model_name="DETR")
                            self.detr_processor.out.write(detr_frame)

                    pbar.update(len(frames_batch))
                    frames_batch = []

            if processed_frames < self.total_frames:
                pbar.update(self.total_frames - processed_frames)

        if self.yolo_processor:
            self.yolo_processor.cleanup()
            print(f"YOLO output saved as {self.yolo_processor.output_path}")
        if self.detr_processor:
            self.detr_processor.cleanup()
            print(f"DETR output saved as {self.detr_processor.output_path}")
