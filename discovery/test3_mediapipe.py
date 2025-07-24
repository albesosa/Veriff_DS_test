import cv2
import mediapipe as mp

# Initialize Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

# Video loop
video_path = "../data/veriff19.mp4"
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert BGR to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_detection.process(rgb_frame)

    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape
            x, y = int(bboxC.xmin * w), int(bboxC.ymin * h)
            width, height = int(bboxC.width * w), int(bboxC.height * h)

            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)

    cv2.imshow('MediaPipe Face Detection', frame)
    print(f"Detected {len(results.detections) if results.detections else 0} faces.")
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
