from flask import Flask, render_template, Response
from ultralytics import YOLO
import cv2
import numpy as np

app = Flask(__name__)

# YOLOv8 modelini yükle
MODEL_PATH = 'C:/Users/ibrah/OneDrive/Masaüstü/ortak/isaretdili_yolo8/yolo8_trained_model.pt'  # Model dosyanızın tam yolu
model = YOLO(MODEL_PATH)

# Kamerayı başlat
camera = cv2.VideoCapture(0)  # 0, varsayılan web kamerasını temsil eder

def generate_frames():
    while True:
        # Kameradan bir kare yakala
        success, frame = camera.read()
        if not success:
            break

        # Model tahmini için görüntüyü işleme
        results = model(frame)

        # Tahmin edilen kelimeleri al
        labels = []
        for box in results[0].boxes:  # Tespit edilen kutulara erişim
            class_id = int(box.cls.cpu().numpy())  # Sınıf ID'sini al
            labels.append(model.names[class_id])  # Sınıf adını ekle

        # Tespit edilen kelimeleri ekrana yazdır (çerçevenin üstüne)
        for label in labels:
            cv2.putText(frame, label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Görüntüyü web sayfasında göstermek için encode et
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # HTTP üzerinden görüntü gönder
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')  # Canlı kamera sayfası

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/isaret-dili')
def detection():
    return render_template('detection.html')  # detection.html dosyasını yükler

@app.route('/egitim')
def egitim():
    return render_template('egitim.html')

@app.route('/sozluk')
def sozluk():
    return render_template('sozluk.html')


if __name__ == '__main__':
    app.run(debug=True)
