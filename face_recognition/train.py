import cv2
import numpy as np
import os

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier('haarcascade.xml')

faces = []
labels = []
label_map = {}
label_id = 0

dataset_path = "dataset"
for user in os.listdir(dataset_path):
    user_path = os.path.join(dataset_path, user)
    if not os.path.isdir(user_path):
        continue

    label_map[label_id] = user

    for image_name in os.listdir(user_path):
        img_path = os.path.join(user_path, image_name)
        gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        faces.append(gray)
        labels.append(label_id)

    label_id += 1

recognizer.train(faces, np.array(labels))
recognizer.write('trainer.yml')

with open('labels.txt', 'w') as f:
    for k, v in label_map.items():
        f.write(f"{k}:{v}\n")

print("Training complete!")