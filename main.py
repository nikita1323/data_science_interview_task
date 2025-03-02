"""Entry point for the person detection pipeline with command-line interface."""

import argparse
from pipeline import Pipeline


def restricted_float(val):
    """Parse restricted float in range [0.0, 1.0]"""
    val = float(val)
    if val < 0.0 or val > 1.0:
        raise argparse.ArgumentTypeError(f"{val} not in range [0.0, 1.0]")
    return val


def parse_args():
    """Parses command-line arguments for configuring the pipeline.

    Returns:
        argparse.Namespace: Parsed arguments with pipeline configuration.

    Raises:
        argparse.ArgumentError: If required arguments are missing or invalid.
    """
    parser = argparse.ArgumentParser(
        description="Person detection pipeline with YOLO and DETR")
    parser.add_argument("--input_path", type=str,
                        required=True, help="Path to input video file")
    parser.add_argument("--mode", type=str,
                        choices=[
                            "yolo_only",
                            "detr_only",
                            "all"
                        ], default="all",
                        help="Inference mode: 'yolo_only', 'detr_only' or 'all'")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for both YOLO and DETR inference")

    # YOLO arguments
    parser.add_argument("--yolo_output_path", type=str, default="output/yolo12n.mp4",
                        help="Path for YOLO output video")
    parser.add_argument("--yolo_model_name", type=str,
                        choices=[
                            "yolo12n.pt",
                            "yolo12s.pt",
                            "yolo12m.pt",
                            "yolo12l.pt",
                            "yolo12x.pt"
                        ], default="yolo12n.pt",
                        help="YOLO model name or path")
    parser.add_argument("--yolo_conf_threshold", type=restricted_float, default=0.25,
                        help="Confidence threshold for YOLO")
    parser.add_argument("--yolo_iou_threshold", type=restricted_float, default=0.85,
                        help="IoU threshold for YOLO NMS")
    parser.add_argument("--yolo_if_half", action="store_true",
                        help="Use FP16 (half-precision) for YOLO")
    parser.add_argument("--yolo_if_verbose", action="store_true",
                        help="Enable verbose output for YOLO")

    # DETR arguments
    parser.add_argument("--detr_output_path", type=str, default="output/detr-resnet-50.mp4", 
                        help="Path for DETR output video")
    parser.add_argument("--detr_model_name", type=str,
                        choices=[
                            "facebook/detr-resnet-50",
                            "facebook/detr-resnet-50-dc5",
                            "facebook/detr-resnet-101",
                            "facebook/detr-resnet-101-dc5"
                        ], default="facebook/detr-resnet-50",
                        help="DETR model name from Hugging Face")
    parser.add_argument("--detr_conf_threshold", type=restricted_float, default=0.5,
                        help="Confidence threshold for DETR")
    parser.add_argument("--detr_iou_threshold", type=restricted_float, default=0.1,
                        help="IoU threshold for DETR NMS")
    parser.add_argument("--detr_if_verbose", action="store_true",
                        help="Enable verbose output for DETR")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    pipeline = Pipeline(args)
    pipeline.run()
