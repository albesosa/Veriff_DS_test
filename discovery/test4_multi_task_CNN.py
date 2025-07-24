import cv2
from mtcnn import MTCNN

# Initialize MTCNN detector
detector = MTCNN()

# Open video (0 = webcam or use path to video file)
video_path = "../data/veriff16.mp4"
cap = cv2.VideoCapture(video_path)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect faces in the current frame
    results = detector.detect_faces(frame)

    # Draw bounding boxes
    for result in results:
        x, y, w, h = result['box']
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('MTCNN Face Detection', frame)
    print(f"Detected {len(results)} faces.")

    # Exit with 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
