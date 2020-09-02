import cv2
import sys


cascPath = "/home/mickey/PycharmProjects/face3/venv/lib/python3.6/site-packages/cv2/data/haarcascade_frontalface_default.xml"
eyecascPath = "/home/mickey/PycharmProjects/face3/venv/lib/python3.6/site-packages/cv2/data/haarcascade_eye.xml"

faceCascade = cv2.CascadeClassifier(cascPath)
eyeCascade = cv2.CascadeClassifier(eyecascPath)

video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.1, 4)
    eyes = eyeCascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in eyes:
        for(a,b,c,d) in faces:
            if(x > a and x < a+c):
                if(y > b and y < b+d):
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    for(x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)

    cv2.imshow('Video', frame)

    if(cv2.waitKey(1) & 0xFF == ord('q')):
        break

video_capture.release()
cv2.destroyAllWindows()