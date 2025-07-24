import cv2
import pandas as pd

# Load Haar Cascade face detector from opencv
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

video_path = "../data/veriff19.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error opening video")
    exit()

results = []
frame_number = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_number += 1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    total_eyes = 0
    filtered_faces = 0

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 5)

        total_eyes += len(eyes)

        if len(eyes) >= 1:
            filtered_faces += 1
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    results.append({
        "frame": frame_number,
        "faces_detected": len(faces),
        "filtered_faces_with_eyes": filtered_faces,
        "eyes_detected": total_eyes
    })

    # Print detection results
    print(f"Frame {frame_number}: Faces detected = {len(faces)}, Filtered faces (with eyes) = {filtered_faces}, Eyes detected = {total_eyes}")

    # Show frame
    cv2.imshow("Video", frame)

    # Exit with 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

df = pd.DataFrame(results)
print(df)