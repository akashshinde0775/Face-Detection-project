import cv2
import pickle
import numpy as np
import os

video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  # Removed 'data/'

faces_data = []
i = 0

name = input("Enter Your Name: ")

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w, :]
        resized_img = cv2.resize(crop_img, (50, 50))
        if len(faces_data) <= 50 and i % 10 == 0:  # Changed 100 to 50 here
            faces_data.append(resized_img)
        i += 1
        cv2.putText(frame, str(len(faces_data)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 1)
    cv2.imshow("Frame", frame)
    k = cv2.waitKey(1)
    if k == ord('q') or len(faces_data) == 50:  # Changed 100 to 50 here
        break

video.release()
cv2.destroyAllWindows()

faces_data = np.asarray(faces_data)
faces_data = faces_data.reshape(50, -1)  # Changed 100 to 50 here

if 'names.pkl' not in os.listdir():
    names = [name] * 50  # Changed 100 to 50 here
    with open('names.pkl', 'wb') as f:
        pickle.dump(names, f)
else:
    with open('names.pkl', 'rb') as f:
        names = pickle.load(f)
    names = names + [name] * 50  # Changed 100 to 50 here
    with open('names.pkl', 'wb') as f:
        pickle.dump(names, f)

if 'faces_data.pkl' not in os.listdir():
    with open('faces_data.pkl', 'wb') as f:
        pickle.dump(faces_data, f)
else:
    with open('faces_data.pkl', 'rb') as f:
        faces = pickle.load(f)
    faces = np.append(faces, faces_data, axis=0)
    with open('faces_data.pkl', 'wb') as f:
        pickle.dump(faces, f)
