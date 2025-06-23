import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Load the trained mask detection model
model = load_model("model/mask_detector_model.h5")

labels = ["Mask", "No Mask"]
colors = [(0, 255, 0), (0, 0, 255)]  # Green for Mask, Red for No Mask

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        resized = cv2.resize(face_img, (100, 100))  # Same size used during training
        normalized = resized / 255.0
        reshaped = np.reshape(normalized, (1, 100, 100, 3))

        # Predict mask/no mask
        result = model.predict(reshaped)
        label_idx = np.argmax(result)
        label = labels[label_idx]
        color = colors[label_idx]

        # Draw rectangle and label
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Display the result
    cv2.imshow("Mask Detection", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
