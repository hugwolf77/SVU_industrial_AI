# 필요 모듈을 import 합니다.
import cv2
import mediapipe as mp
import time

mpHands = mp.solutions.hands        # mediapipe로 부터 hands모델을 불러옵니다.
hands = mpHands.Hands()             # hands모델로 부터 Hands라는 손을 추적하는 클래스를 불러옵니다.
mpDraw = mp.solutions.drawing_utils # mediapipe로 부터 결과 값을 image에 그려 넣는 클래스를 불러옵니다.
mp_drawing_styles = mp.solutions.drawing_styles

count_time = 0                      # fps를 계산할 변수를 선언합니다.
elapsed_time = 0

# 분석할 동영상 이름
file_name = 0
# 결롸로 저장할 동영상 이름
save_result = None #'/results/handpose.mp4'

 # 분석 동영상 mp4 읽어오기
cap = cv2.VideoCapture(file_name)
idx = 0

# 재생할 파일의 넓이와 높이 읽어 두기
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(save_result, fourcc, fps, (int(width), int(height)),True)
print("재생할 파일 넓이, 높이 : %d, %d"%(width, height))

if not cap.isOpened:
    print('--(!)Error opening video capture')
    exit(0)

while (cap.isOpened()):
    start_time = time.time()
    idx += 1
    success, frame = cap.read()                         # 동영상의 frame을 읽어 옵니다.

    if frame is None:                                   # 동영상의 frame이 없으면 분석을 끝냅니다..
      # close the video file pointers
      cap.release()
      # close the writer point
      writer.release()
      print('--(!) No captured frame -- Break!')
      print("elapsed time {:.3f} seconds".format(elapsed_time))
      break
    if success == False:
        break;

    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)       # BGR 포맷에서 RGB 포맷으로 이미지를 변환합니다.
    results = hands.process(frameRGB)                     # hands클래스의 process명령으로 변환한 이미지를 분석합니다.
    # print(results.multi_hand_landmarks)

#    if not results.multi_hand_landmarks:
#      continue

    if results.multi_hand_landmarks:                    # 읽어 드린 결과 중 손의 landmark 결과가 있는 경우 진행합니다.
        for handLMK in results.multi_hand_landmarks:    # 읽어 드린 결과에서 찾은 손의 수(사용자 지정 기본 2)만큼 순환 진행
            for id, lmk in enumerate(handLMK.landmark): # 읽어 드린 손의 결과의 landmark 수만큼 순환 진행
                # print(id, lmk)
                h, w, c = frame.shape                                     # 원본 이미지의 크기
                cx, cy = int(lmk.x*w), int(lmk.y*h)                     # 해당 landmark의 좌표값을 곱해서 표시할 위치 확인
                # print(id, cx, cy)                                       # 해당 landmark의 위치 표시
                if id==0:                                               # 표시하고 싶은 손의 landmark 위치 지정 현재는 손바닥 중심(0번)
                    cv2.circle(frame,(cx,cy),15,(255,0,255),cv2.FILLED )  # 해당 위치에 원 표시
            mpDraw.draw_landmarks(frame, handLMK, mpHands.HAND_CONNECTIONS
                                  ,mp_drawing_styles.get_default_hand_landmarks_style()
                                  ,mp_drawing_styles.get_default_hand_connections_style()) # mediapipe  draw 클래스로 landmark 위치점들 연결선 그리기
    elapsed_time = time.time()
    fps = 1/(elapsed_time-count_time)                                      # 프레임 처리 시간 계산
    count_time = elapsed_time

    cv2.putText(frame,str(int(fps)),(20,140),cv2.FONT_HERSHEY_PLAIN        # 프레임 처리시간 표시
                , 3, (255,0,255),3)

    process_time = time.time() - start_time
    elapsed_time += process_time   # 총 경과시간 누적
    print("=== A idx {:d} frame took {:.3f} seconds".format(idx,process_time))

    # frame의 저장
    if save_result is not None:
        writer.write(frame)
        
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) == ord("q"):
        break
    
cap.release()
writer.release()
cv2.destroyAllWindows()