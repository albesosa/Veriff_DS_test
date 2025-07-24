import cv2
import pandas as pd
from src.utils import setup_logging

logger = setup_logging()


class VideoSpoofProcessor:
    """
    Video processing module for spoof detection.
    This module processes video files to detect faces, classify them as real or fake,
    and track them across frames. It uses a face detector, spoof detector, and face tracker.
    It provides a unified interface for processing videos and saving results.

    Attributes:
        face_detector: An instance of a face detection class.
        spoof_detector: An instance of a spoof detection class.
        face_tracker: An instance of a face tracking class.
        video_path: Path to the video file to be processed.
        thr: Threshold for spoof detection classification.
        show: Boolean flag to indicate whether to display the video while processing.
        results: List to store results for each frame.
    Methods:
        process_video: Processes the video file, detects faces, classifies them, and tracks them.
        _save_results: Saves the results to a CSV file.
        final_label_prediction: Predicts if the video contains frames with more than one real face.
    Usage:
        processor = VideoSpoofProcessor(face_detector, spoof_detector, face_tracker, video_path, thr=0.25, show=False)
        processor.process_video()
        processor._save_results()
        predicted_label = processor.final_label_prediction(video_path_name)
    """
    def __init__(self, face_detector, spoof_detector, face_tracker, video_path, thr=0.25, show=False):
        self.face_detector = face_detector
        self.spoof_detector = spoof_detector
        self.tracker = face_tracker
        self.show = show
        self.results = []
        self.thr = thr
        self.video_path = video_path

    def process_video(self):
        cap = cv2.VideoCapture(self.video_path)
        frame_number = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            logger.info(f"Detecting faces in frame: {frame_number + 1} of {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}")
            raw_faces = self.face_detector.detect_faces(frame)  # [(x, y, w, h), ...]
            logger.info("Tracking faces...")
            tracked_faces = self.tracker.update(raw_faces)  # [(x, y, w, h, track_id), ...]

            frame_data = []
            for (x, y, w, h, track_id) in tracked_faces:
                x2, y2 = x + w, y + h
                face = frame[y:y+h, x:x+w]

                if face.size == 0:
                    continue
                logger.info("Classifying face for spoof detection...")
                score = self.spoof_detector.predict(face)
                label = "REAL" if score > self.thr else "FAKE"

                frame_data.append({
                    "frame_number": frame_number,
                    "track_id": track_id,
                    "x": x, "y": y, "w": w, "h": h,
                    "spoof_score": round(score, 4),
                    "spoof_label": label
                })

                if self.show:
                    color = (0, 255, 0) if label == "REAL" else (0, 0, 255)
                    cv2.rectangle(frame, (x, y), (x2, y2), color, 2)
                    cv2.putText(frame, f"ID {track_id} {label} ({score:.2f})",
                                (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            self.results.extend(frame_data)
            frame_number += 1

            if self.show:
                cv2.imshow("Frame", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

        cap.release()
        if self.show:
            cv2.destroyAllWindows()

    def _save_results(self):
        face_detector_method = self.face_detector.method
        spoof_detector_method = self.spoof_detector.method
        face_tracker_method = self.tracker.method
        models_name = f"{face_detector_method}_{spoof_detector_method}_{face_tracker_method}"
        video_name = self.video_path.split("/")[-1].split(".")[0]
        df = pd.DataFrame(self.results)
        df.to_csv(f"output_data/results_{video_name}_{models_name}.csv", index=False)

    def final_label_prediction(self, video_path_name):
        """Predicts if a video contains frames with more than one real face.
        Returns:
            predicted_label (int):
                1 if there are any frames with more than one real face detected,
                0 if none or if the results file is empty.
        """

        results_df = pd.DataFrame(self.results)

        if video_path_name:
            # Load CSV file
            try:
                results_df = pd.read_csv(video_path_name)
                # Return 0 if DataFrame is empty and has no columns (e.g., CSV exists but is completely empty)
                if results_df.empty and results_df.columns.size == 0:
                    return 0
            except pd.errors.EmptyDataError:
                # Return 0 if CSV is completely empty (no content at all)
                return 0
        elif results_df.empty:
            # If results_df is empty, return 0 immediately
            return 0

        # Calculate number of frames labeled FAKE or REAL for each track_id
        tracking_consistency = results_df.groupby("track_id").agg(
            n_labeled_fake=('spoof_label', lambda x: (x == "FAKE").sum()),
            n_labeled_real=('spoof_label', lambda x: (x == "REAL").sum())
        )

        # Compute probability of each track being fake
        tracking_consistency['prob_real_fake'] = tracking_consistency['n_labeled_fake'] / (
                tracking_consistency['n_labeled_fake'] + tracking_consistency['n_labeled_real']
        )

        # Assign corrected spoof label by applying a threshold
        tracking_consistency['spoof_label_corrected'] = tracking_consistency['prob_real_fake'].apply(
            lambda x: "FAKE" if x > 0.3 else "REAL"
        )

        # Merge corrected labels back into main results DataFrame
        results_df = results_df.merge(tracking_consistency, on="track_id", how="left")

        # Group by frame and count real and spoofed faces
        grouped = results_df.groupby("frame_number").agg(
            total_faces=('frame_number', 'count'),
            spoofed_faces=('spoof_label_corrected', lambda x: (x == "FAKE").sum())
        ).reset_index()

        grouped['total_real_faces'] = grouped['total_faces'] - grouped['spoofed_faces']

        # Select frames with more than five faces detected
        df_more_than_two_faces = grouped[grouped['total_faces'] > 5]

        # Select frames with more than one real face
        df_more_than_one_real_face = grouped[grouped['total_real_faces'] > 1]

        # Predict label based on real face counts
        if len(df_more_than_two_faces) > 10:
            # If there are more than 10 frames with more than five faces, I consider it a strong
            # indication of multiple persons
            predicted_label = 1
        elif len(df_more_than_one_real_face) > 0:
            # If there are any frames with more than one real face, I consider it a strong indication
            # of multiple persons
            predicted_label = 1
        else:
            predicted_label = 0

        return predicted_label
