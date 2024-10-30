
import numpy
import cv2
print(cv2.__version__)
import random
import matplotlib.pyplot as plt


cascade_pre = './pre/haarcascade_frontalface_alt2.xml'
cascade = cv2.CascadeClassifier(cascade_pre)

cap = cv2.VideoCapture(2)


while True:
    ret, frame = cap.read()
    
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # trans gray
    face_list = cascade.detectMultiScale(gray, minSize = (50,50))
        
          # 검출 결과 화면 표시
    for (x, y, w, h) in face_list:
        c = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        cv2.rectangle(frame, (x, y, w, h), c, 3)
        
    cv2.imshow('frame', frame)
    if cv2.waitKey(10) == 27:
        break

cv2.destroyAllWindows()