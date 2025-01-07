import cv2

file_path = '/home/augustine77/Lab_2024/p01_base/SVU_industrial_AI/Vision/OpenCV/임재범 - 비상 [불후의 명곡2 전설을 노래하다Immortal Songs 2] | KBS 220903 방송.mp4'

cascade_pre = './face/pre/haarcascade_frontalface_alt2.xml'
cascade = cv2.CascadeClassifier(cascade_pre)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # trans gray
    face_list = cascade.detectMultiScale(gray, minSize = (20,20))
          
    # 검출 결과 화면 표시
    for (x, y, w, h) in face_list:
        # c = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        c = (0,255,0)
        cv2.rectangle(frame, (x, y, w, h), c, 3)

    cv2.imshow('frame', frame)
    if cv2.waitKey(10) == 27:
            break