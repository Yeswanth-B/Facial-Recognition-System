import cv2
import os
from datetime import datetime

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer.yml')
detector = cv2.CascadeClassifier('haarcascade.xml')

label_map = {}
with open('labels.txt') as f:
    for line in f:
        k, v = line.strip().split(":")
        label_map[int(k)] = v

def mark_attendance(name):
    date_str = datetime.now().strftime('%Y-%m-%d')
    log_file = f'attendance_{date_str}.csv'
    if not os.path.exists(log_file):
        with open(log_file, 'w') as f:
            f.write('Name,Time\n')
    with open(log_file, 'r+') as f:
        entries = f.read()
        now = datetime.now().strftime('%H:%M:%S')
        if name not in entries:
            f.write(f'{name},{now}\n')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_img = gray[y:y+h, x:x+w]
        id_, conf = recognizer.predict(face_img)
        name = label_map.get(id_, "Unknown")

        color = (0, 255, 0) if conf < 60 else (0, 0, 255)
        label = f"{name} ({round(conf, 2)})" if conf < 60 else "Unknown"

        if conf < 60:
            mark_attendance(name)

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow('Face Recognition - Attendance', frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()