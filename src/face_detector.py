import cv2
import numpy as np
import os
import mediapipe as mp

class FaceDetector:
    """
    Face detection module using YOLOv3 Tiny or MediaPipe.
    This module provides a unified interface for face detection using different methods.
    It supports YOLOv3 Tiny for high-performance detection and MediaPipe for lightweight applications.
    New face detection methods can be easily integrated by implementing the `detect_faces` method.
    Attributes:
        method (str): The face detection method to use ('yolov3' or 'mediapipe').
        detector: An instance of the selected face detection class.
    Methods:
        detect_faces(frame): Detects faces in the given frame using the selected method.
    Usage:
        face_detector = FaceDetector(method="yolov3")
        faces_detected = face_detector.detect_faces(frame)
    """
    def __init__(self, method="yolov3"):
        self.method = method
        if self.method == "yolov3":
            self.detector = YoloV3TinyFace()
        elif self.method == "mediapipe":
            self.detector = MediaPipeFace()
        else:
            raise ValueError(f"Unsupported face detection method: {self.method}")

    def detect_faces(self, frame):
        return self.detector.detect_faces(frame)


class YoloV3TinyFace:
    def __init__(self, confidence_threshold=0.5):
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # path to src/
        cfg_path = os.path.join(BASE_DIR, "..", "models", "yolo_face", "yolov3-tiny-face.cfg")
        weights_path = os.path.join(BASE_DIR, "..", "models", "yolo_face", "yolov3-tiny-face.weights")
        # normalize paths
        cfg_path = os.path.abspath(cfg_path)
        weights_path = os.path.abspath(weights_path)

        # Initialize YOLOv3 Tiny face detector
        self.net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        layer_names = self.net.getLayerNames()
        self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers().flatten()]
        self.conf_threshold = confidence_threshold

    def detect_faces(self, frame):
        height, width = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        detections = self.net.forward(self.output_layers)

        boxes, confidences = [], []
        for output in detections:
            for detection in output:
                confidence = detection[4]
                if confidence > self.conf_threshold:
                    cx = int(detection[0] * width)
                    cy = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = max(int(cx - w / 2), 0)
                    y = max(int(cy - h / 2), 0)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))

        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, 0.4)
        # Safely normalize indices
        if indices is not None and len(indices) > 0:
            # Handle tuple or array output
            if isinstance(indices, tuple):
                if len(indices) > 0 and hasattr(indices[0], 'flatten'):
                    indices = indices[0].flatten()
                else:
                    indices = np.array([])  # Empty result
            else:
                indices = np.array(indices).flatten()
            final_boxes = [boxes[i] for i in indices]
        else:
            final_boxes = []

        return final_boxes


class MediaPipeFace:
    def __init__(self, confidence_threshold=0.5):
        self.confidence_threshold = confidence_threshold
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=confidence_threshold
        )

    def detect_faces(self, frame):
        height, width = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_frame)

        boxes = []
        if results.detections:
            for detection in results.detections:
                if detection.score[0] >= self.confidence_threshold:
                    bboxc = detection.location_data.relative_bounding_box
                    x = int(bboxc.xmin * width)
                    y = int(bboxc.ymin * height)
                    w = int(bboxc.width * width)
                    h = int(bboxc.height * height)
                    boxes.append([x, y, w, h])

        return boxes
