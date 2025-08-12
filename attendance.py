import cv2
import numpy as np
import face_recognition
import pickle
from datetime import datetime
import pandas as pd
import os

# Load encodings and names
with open('encodings.pkl', 'rb') as f:
    encodeListKnown, classNames = pickle.load(f)

# Initialize webcam
cap = cv2.VideoCapture(0)

# Load or create attendance CSV file
attendance_file = 'Attendance.csv'
if os.path.exists(attendance_file):
    df = pd.read_csv(attendance_file)
else:
    df = pd.DataFrame(columns=['Name', 'Time'])

def markAttendance(name):
    global df  # Declare global BEFORE using df
    now = datetime.now()
    dtString = now.strftime('%Y-%m-%d %H:%M:%S')
    today = now.strftime('%Y-%m-%d')

    # Check if already present in attendance for the day
    if not ((df['Name'] == name) & (df['Time'].str.contains(today))).any():
        df = pd.concat([df, pd.DataFrame({'Name': [name], 'Time': [dtString]})], ignore_index=True)
        df.to_csv(attendance_file, index=False)
        print(f"Marked attendance for {name} at {dtString}")

while True:
    success, img = cap.read()
    if not success:
        break
    imgS = cv2.resize(img, (0,0), fx=0.25, fy=0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4

            cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.rectangle(img, (x1, y2-35), (x2,y2), (0,255,0), cv2.FILLED)
            cv2.putText(img, name, (x1+6,y2-6), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            markAttendance(name)

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()