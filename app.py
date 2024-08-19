from flask import Flask, render_template, Response
import cv2
import threading
from ultralytics import YOLO
import pyttsx3

app = Flask(__name__)
model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

announced_objects = set()

def speak(name):
    tts = pyttsx3.init()
    tts.say(name)
    tts.runAndWait()

def generate_frames():
    global announced_objects
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        else:
            results = model(frame)
            for result in results:
                if hasattr(result, 'names') and result.names:
                    class_ids = result.boxes.cls.cpu().numpy()
                    boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2 format
                    confidences = result.boxes.conf.cpu().numpy()

                    for i, class_id in enumerate(class_ids):
                        name = result.names[int(class_id)] if int(class_id) in result.names else "Unknown"
                        box = boxes[i]
                        confidence = confidences[i]

                        if confidence >= 0.8:
                            x1, y1, x2, y2 = box
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                            cv2.putText(frame, f"{name} ({confidence:.2f})", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                            if name not in announced_objects:
                                announced_objects.add(name)
                                threading.Thread(target=speak, args=(name,)).start()

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
