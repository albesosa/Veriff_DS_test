import cv2
import numpy as np

# Paths to your YOLO files
weights_path = "../models/yolo_face/yolov3-tiny-face.weights"
config_path = "../models/yolo_face/yolov3-tiny-face.cfg"

# Load YOLO
net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)  # or DNN_TARGET_CUDA if you have GPU

# Load video file or webcam
video_path = "../data/veriff19.mp4"
cap = cv2.VideoCapture(video_path)  # Or 0 for webcam

# Get output layer names of YOLO
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width = frame.shape[:2]

    # Create a blob and do a forward pass
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward(output_layers)

    boxes = []
    confidences = []

    for output in detections:
        for detection in output:
            scores = detection[5:]  # No classes here, but convention kept
            confidence = detection[4]  # objectness score

            if confidence > 0.5:  # Threshold
                # YOLO outputs center_x, center_y, width, height (all normalized)
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    if indices is not None and len(indices) > 0:
        # indices might be a tuple, so convert to list of ints
        indices = indices.flatten() if hasattr(indices, 'flatten') else [i[0] for i in indices]
        for i in indices:
            x, y, w, h = boxes[i]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Face {confidences[i]:.2f}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Face Detection - YOLOv3-tiny", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC key to stop
        break

cap.release()
cv2.destroyAllWindows()
