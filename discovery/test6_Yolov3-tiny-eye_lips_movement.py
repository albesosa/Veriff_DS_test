import cv2
import numpy as np
import mediapipe as mp

# === Load YOLO face detector ===
weights_path = "../models/yolo_face/yolov3-tiny-face.weights"
config_path = "../models/yolo_face/yolov3-tiny-face.cfg"
net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# === Initialize MediaPipe Face Mesh ===
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
drawing_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=1, circle_radius=1)

# === Video ===
cap = cv2.VideoCapture("../data/veriff1.mp4")

def eye_aspect_ratio(landmarks, eye_indices):
    # Compute EAR
    p1 = np.array(landmarks[eye_indices[1]])
    p2 = np.array(landmarks[eye_indices[5]])
    p3 = np.array(landmarks[eye_indices[2]])
    p4 = np.array(landmarks[eye_indices[4]])
    p5 = np.array(landmarks[eye_indices[0]])
    p6 = np.array(landmarks[eye_indices[3]])
    # vertical
    vertical1 = np.linalg.norm(p2 - p4)
    vertical2 = np.linalg.norm(p3 - p5)
    # horizontal
    horizontal = np.linalg.norm(p1 - p6)
    return (vertical1 + vertical2) / (2.0 * horizontal)

def lip_distance(landmarks):
    top = np.array(landmarks[13])
    bottom = np.array(landmarks[14])
    return np.linalg.norm(top - bottom)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward(output_layers)

    boxes = []
    confidences = []

    for output in detections:
        for detection in output:
            confidence = detection[4]
            if confidence > 0.5:
                cx = int(detection[0] * width)
                cy = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(cx - w / 2)
                y = int(cy - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    if indices is not None and len(indices) > 0:
        indices = indices.flatten()
        for i in indices:
            x, y, w, h = boxes[i]

            side_len = max(w, h)
            cx = x + w // 2
            cy = y + h // 2

            # Calculate initial top-left corner for the square box
            x1 = cx - side_len // 2
            y1 = cy - side_len // 2
            x2 = x1 + side_len
            y2 = y1 + side_len

            # Image dimensions
            img_h, img_w = frame.shape[:2]

            # Shift the box if it goes beyond the left or top edge
            if x1 < 0:
                x2 += -x1  # increase x2 by the overflow amount
                x1 = 0
            if y1 < 0:
                y2 += -y1  # increase y2 by the overflow amount
                y1 = 0

            # Shift the box if it goes beyond the right or bottom edge
            if x2 > img_w:
                overflow = x2 - img_w
                x1 -= overflow
                x2 = img_w
                if x1 < 0:
                    x1 = 0  # clamp again if shifting pushed it negative
            if y2 > img_h:
                overflow = y2 - img_h
                y1 -= overflow
                y2 = img_h
                if y1 < 0:
                    y1 = 0  # clamp again if shifting pushed it negative

            # Convert to int for indexing
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            # Sanity check to ensure the box is square
            assert (x2 - x1) == (y2 - y1), f"Bounding box not square: {(x2 - x1)}x{(y2 - y1)}"

            face_roi = frame[y1:y2, x1:x2]

            # Skip if ROI is empty (invalid crop)
            if face_roi.size == 0:
                print(f"Warning: Empty face ROI at index {i}, box=({x}, {y}, {w}, {h})")
                continue
            rgb_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_face)

            label = "STATIC"
            color = (0, 255, 0)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    landmarks = []
                    for lm in face_landmarks.landmark:
                        px = int(lm.x * w)
                        py = int(lm.y * h)
                        landmarks.append((px, py))

                    # Blink detection
                    left_eye_indices = [33, 160, 158, 133, 153, 144]
                    right_eye_indices = [362, 385, 387, 263, 373, 380]
                    ear_left = eye_aspect_ratio(landmarks, left_eye_indices)
                    ear_right = eye_aspect_ratio(landmarks, right_eye_indices)
                    ear_avg = (ear_left + ear_right) / 2

                    # Lip movement
                    lip_dist = lip_distance(landmarks)

                    # Thresholds
                    if ear_avg < 0.2:
                        label = "BLINK"
                        color = (0, 255, 255)
                    if lip_dist > 15:
                        label = "LIP MOVEMENT"
                        color = (255, 0, 255)

            # Draw results
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("YOLO + Liveness", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
