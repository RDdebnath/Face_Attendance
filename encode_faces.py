import cv2
import face_recognition
import os
import numpy as np
import pickle

path = r"C:\Users\debna\OneDrive\Desktop\Face_AttendanceSystem\Images"
images = []
classNames = []
myList = os.listdir(path)

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])  # filename without extension

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)
        if encode:
            encodeList.append(encode[0])
    return encodeList

print("Encoding faces...")
encodeListKnown = findEncodings(images)
print('Encoding complete')

# Save encodings and names for later use
with open('encodings.pkl', 'wb') as f:
    pickle.dump((encodeListKnown, classNames), f)