import os
import cv2
import numpy as np
from flask import Flask, render_template, request, Response
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

model = load_model("model/mask_detector_model.h5")
labels = ["Mask", "No Mask"]
colors = [(0, 255, 0), (0, 0, 255)]

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    image_path = None

    if request.method == "POST":
        file = request.files["image"]
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            img = cv2.imread(filepath)
            resized = cv2.resize(img, (100, 100))
            normalized = resized / 255.0
            reshaped = np.reshape(normalized, (1, 100, 100, 3))

            prediction = model.predict(reshaped)
            label_idx = np.argmax(prediction)
            result = labels[label_idx]
            image_path = "/" + filepath

    return render_template("index.html", result=result, image_path=image_path)


@app.route("/webcam")
def webcam():
    return render_template("webcam.html")


def generate_frames():
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)

        for (x, y, w, h) in faces:
            face_img = frame[y:y + h, x:x + w]
            resized = cv2.resize(face_img, (100, 100))
            normalized = resized / 255.0
            reshaped = np.reshape(normalized, (1, 100, 100, 3))

            prediction = model.predict(reshaped)
            label_idx = np.argmax(prediction)
            label = labels[label_idx]
            color = colors[label_idx]

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)
