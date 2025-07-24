import cv2
import torch
from models.DeepPixBisS_anti_spoofing.Model import DeePixBiS  # from DeepPixBiS repo
from torchvision import transforms

# === Load YOLO ===
net = cv2.dnn.readNetFromDarknet("../yolo_face/yolov3-tiny-face.cfg", "../yolo_face/yolov3-tiny-face.weights")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# === Load DeepPixBiS model ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DeePixBiS().to(device)  # From Model.py
model.load_state_dict(torch.load("../models/DeepPixBisS_anti_spoofing/DeePixBiS.pth", map_location=device))
model.eval().to(device)

# === Image preprocessing ===
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# === Video capture ===
video_path = "../data/veriff2.mp4"
cap = cv2.VideoCapture(video_path)

while True:
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
                x = int(cx - w/2)
                y = int(cy - h/2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    if indices is not None and len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]

            # Make bounding box square and safe
            side = max(w, h)
            x = max(x + w//2 - side//2, 0)
            y = max(y + h//2 - side//2, 0)
            x2 = min(x + side, width)
            y2 = min(y + side, height)

            face = frame[y:y2, x:x2]
            if face.size == 0:
                continue

            # DeepPixBiS anti-spoofing
            input_tensor = transform(face).unsqueeze(0).to(device)
            with torch.no_grad():
                mask, binary = model(input_tensor)
                prediction = binary.item()

            label = "REAL" if prediction > 0.4 else "FAKE"
            color = (0, 255, 0) if label == "REAL" else (0, 0, 255)

            cv2.rectangle(frame, (x, y), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} ({prediction:.2f})", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("YOLO + DeepPixBiS", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
