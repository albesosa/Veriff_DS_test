from src.face_detector import FaceDetector
from src.spoof_detector import SpoofDetector
from src.face_tracker import FaceTracker
from src.video_processor import VideoSpoofProcessor
import argparse
import os
import sys
import logging
from src.utils import setup_logging


# Supported video formats
VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov')


def process_video(video_path, threshold, show,
                  face_detector_method=None,
                  spoof_detector_method=None,
                  face_tracker_method=None,
                  video_path_name=None):
    logger.info(f"\nProcessing: {video_path}")

    face_detector = FaceDetector(method=face_detector_method)
    spoof_detector = SpoofDetector(method=spoof_detector_method)
    face_tracker = FaceTracker(method=face_tracker_method)

    processor = VideoSpoofProcessor(
        face_detector,
        spoof_detector,
        face_tracker,
        video_path=video_path,
        thr=threshold,
        show=show
    )

    try:
        processor.process_video()
        processor._save_results()
        predicted_label = processor.final_label_prediction(video_path_name)
        logger.info(f"Finished: {os.path.basename(video_path)}\n")
        logger.info(f"Predicted label for {os.path.basename(video_path)}: {predicted_label}")

    except Exception as e:
        logger.info(f"Failed: {os.path.basename(video_path)} | Reason: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video spoof detection")
    parser.add_argument('--data_dir', type=str, default='data', help='Directory containing videos')
    parser.add_argument('--video_to_process', type=str, required=True,
                        help='Use all" to process all videos, or provide a video file path')
    parser.add_argument('--threshold', type=float, default=0.25, help='Spoof detection threshold')
    parser.add_argument('--show', action='store_true', help='Show video while processing')
    parser.add_argument('--face_detector_method', type=str, default='yolov3',
                        help='Face detection method (e.g., yolov3, mediapipe)')
    parser.add_argument('--spoof_detector_method', type=str, default='deeppixbis',
                        help='Spoof detection method (e.g., cdcnpp, deeppixbis)')
    parser.add_argument('--face_tracker_method', type=str, default='sort',
                        help='Face tracking method (e.g., sort)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    args = parser.parse_args()

    log_level = logging.INFO if args.verbose else logging.CRITICAL
    logger = setup_logging(log_level)

    # Handle "al" or "l" for processing all videos
    if args.video_to_process.lower() == "all":
        if not os.path.isdir(args.data_dir):
            logger.info(f"Error: '{args.data_dir}' is not a valid directory.")
            sys.exit(1)

        video_list = [
            os.path.join(args.data_dir, f)
            for f in os.listdir(args.data_dir)
            if f.lower().endswith(VIDEO_EXTENSIONS)
        ]

        if not video_list:
            logger.info(f"No video files found in {args.data_dir}")
            sys.exit(1)
    else:
        if not os.path.isfile(args.video_to_process):
            logger.info(f"Error: File '{args.video_to_process}' not found.")
            sys.exit(1)
        video_list = [args.video_to_process]

    # Process all selected videos
    for video_path_k in video_list:
        process_video(video_path_k, args.threshold, args.show,
                      face_detector_method=args.face_detector_method,
                      spoof_detector_method=args.spoof_detector_method,
                      face_tracker_method=args.face_tracker_method)


# Process a single video
#python -m src.main_video_processing --video_to_process data/veriff6.mp4 --show

# Process all videos in the directory
#python -m src.main_video_processing --video_to_process all --data_dir data