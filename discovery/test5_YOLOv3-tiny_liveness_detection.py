import cv2
import numpy as np

# Paths to YOLOv3-tiny face detection model
weights_path = "../models/yolo_face/yolov3-tiny-face.weights"
config_path = "../models/yolo_face/yolov3-tiny-face.cfg"

# Load YOLO model
net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Load video
video_path = "../data/veriff5.mp4"
cap = cv2.VideoCapture(video_path)

# Get output layer names
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# Store previous face center positions for movement detection
prev_centers = {}

frame_count = 0
face_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width = frame.shape[:2]

    # YOLO preprocessing
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward(output_layers)

    boxes = []
    confidences = []

    for output in detections:
        for detection in output:
            confidence = detection[4]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    current_centers = {}
    movement_detected = False

    if indices is not None and len(indices) > 0:
        indices = indices.flatten() if hasattr(indices, 'flatten') else [i[0] for i in indices]
        for i in indices:
            x, y, w, h = boxes[i]
            cx, cy = x + w // 2, y + h // 2
            current_centers[i] = (cx, cy)

            # Movement detection: compare with previous center
            if i in prev_centers:
                dx = abs(cx - prev_centers[i][0])
                dy = abs(cy - prev_centers[i][1])
                if dx > 3 or dy > 3:
                    movement_detected = True
                    label = "LIVE FACE"
                    color = (0, 255, 255)
                    print(f"[Frame {frame_count}] Face {i}: Movement detected (dx={dx}, dy={dy})")
                else:
                    label = "STATIC FACE"
                    color = (0, 0, 255)
                    print(f"[Frame {frame_count}] Face {i}: No significant movement (dx={dx}, dy={dy})")
            else:
                label = "NEW FACE"
                color = (255, 255, 0)
                print(f"[Frame {frame_count}] Face {i}: New detection (center=({cx},{cy}))")

            # Draw detection and label
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label} ({confidences[i]:.2f})", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    prev_centers = current_centers
    frame_count += 1

    # Display frame
    cv2.imshow("YOLOv3-tiny Face Detection + Liveness", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
