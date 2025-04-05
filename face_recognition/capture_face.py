import cv2
import os

user_name = input("Enter the name of the person: ").strip()
save_path = f'dataset/{user_name}'
os.makedirs(save_path, exist_ok=True)

cap = cv2.VideoCapture(0)
detector = cv2.CascadeClassifier('haarcascade.xml')

count = 0
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        count += 1
        face_img = gray[y:y+h, x:x+w]
        cv2.imwrite(f"{save_path}/{count}.jpg", face_img)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow('Capturing Face', frame)
    if cv2.waitKey(1) == 27 or count >= 30:
        break

cap.release()
cv2.destroyAllWindows()
print(f"Saved {count} images to {save_path}")