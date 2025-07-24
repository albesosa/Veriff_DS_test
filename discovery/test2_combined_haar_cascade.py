import cv2
import pandas as pd
import numpy as np


def rotate_image(image, angle):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h))

def adjust_gamma(image, gamma=1.5):
    invGamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** invGamma * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

# Load Haar Cascade face detector from opencv
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
profile_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")
full_body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_fullbody.xml")


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
    #rotated_img = rotate_image(gray, 15)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    total_eyes = 0
    filtered_faces = 0
    gray_masked = gray.copy()

    # 1. Detect frontal faces (with eyes)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 5)

        total_eyes += len(eyes)

        if len(eyes) >= 1:
            filtered_faces += 1
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        gray_masked[y:y + h, x:x + w] = 0

    # Right-facing detection (default)
    profile_faces_right = profile_face_cascade.detectMultiScale(gray_masked, 1.10, 5)

    # Left-facing detection by flipping
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_clahe = clahe.apply(gray_masked)

    kernel_sharpen = np.array([[0, -1, 0],
                               [-1, 5, -1],
                               [0, -1, 0]])
    sharp = cv2.filter2D(gray_clahe, -1, kernel_sharpen)

    brightened = adjust_gamma(gray, gamma=1.5)

    flipped = cv2.flip(sharp, 1)
    rotated_flipped = rotate_image(flipped, 0)
    profile_faces_left = profile_face_cascade.detectMultiScale(rotated_flipped, 1.1, 5)
    frame_width = rotated_flipped.shape[1]

    for (x, y, w, h) in profile_faces_left:
        flipped_x = frame_width - x - w
        cv2.rectangle(frame, (flipped_x, y), (flipped_x + w, y + h), (0, 255, 255), 2)  # Yellow box

    # 3. Detect full bodies
    bodies = full_body_cascade.detectMultiScale(gray, 1.1, 5)
    for (x, y, w, h) in bodies:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 2)  # Magenta for body

    # Collect results
    results.append({
        "frame": frame_number,
        "faces_detected": len(faces),
        "filtered_faces_with_eyes": filtered_faces,
        "eyes_detected": total_eyes,
        "profile_left_detected": len(profile_faces_left),
        "profile_right_detected": len(profile_faces_right),
        "bodies_detected": len(bodies)
    })

    # Print detection results
    print(f"Frame {frame_number}: Faces detected = {len(faces)},"
          f" Filtered faces (with eyes) = {filtered_faces}, "
          f"Eyes detected = {total_eyes}, "
          f"Profile faces (left) = {len(profile_faces_left)}, "
          f"Profile faces (right) = {len(profile_faces_right)}, "
          f"Bodies detected = {len(bodies)}"
          )

    # Show frame
    cv2.imshow("Video", frame)

    # Exit with 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()





df = pd.DataFrame(results)
print(df)