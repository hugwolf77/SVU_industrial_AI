
import os
import numpy as np
import pandas as pd
import cv2
print(cv2.__version__)
import random
import matplotlib.pyplot as plt
# import dlib
import mediapipe as mp
import time

# mpHands = mp.solutions.hands        # mediapipe로 부터 hands모델을 불러옵니다.
# hands = mpHands.Hands()             # hands모델로 부터 Hands라는 손을 추적하는 클래스를 불러옵니다.
mp_pose = mp.solutions.pose
poses = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_Draw = mp.solutions.drawing_utils # mediapipe로 부터 결과 값을 image에 그려 넣는 클래스를 불러옵니다.
mp_drawing_styles = mp.solutions.drawing_styles

count_time = 0                      # fps를 계산할 변수를 선언합니다.
elapsed_time = 0


def calculate_angle(a,b,c):
    # 각 값을 받아 넘파이 배열로 변형
    a = np.array(a) # 첫번째
    b = np.array(b) # 두번째
    c = np.array(c) # 세번째
    # 라디안을 계산하고 실제 각도로 변경한다.
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    # 180도가 넘으면 360에서 뺀 값을 계산한다.
    if angle >180.0:
        angle = 360-angle
    # 각도를 리턴한다.
    return angle

# 결롸로 저장할 동영상 이름
save_result = None #'/results/handpose.mp4'

# 분석할 동영상 이름
file_name = '/home/augustine77/Lab_2024/p01_base/SVU_industrial_AI/Vision/store/슬로우모션 정면 [이도형프로].mp4'
# '/home/augustine77/Lab_2024/p01_base/SVU_industrial_AI/Vision/store/[골프스윙] 7번아이언 슬로우모션.mp4'
# '/home/augustine77/Lab_2024/p01_base/SVU_industrial_AI/Vision/store/드라이버스윙 피니쉬는 왼발뒷꿈치에 체중싣기.mp4'
# '/home/augustine77/Lab_2024/p01_base/SVU_industrial_AI/Vision/store/슬로우모션 정면 [이도형프로].mp4'
# '/home/augustine77/Lab_2024/p01_base/SVU_industrial_AI/Vision/store/슬로우모션이 정말 멋집니다!!👍👍 KLPGA 황아름 프로 드라이버 스윙. instagram@areum2klpga.mp4'


 # 분석 동영상 mp4 읽어오기
cap = cv2.VideoCapture(file_name)
idx = 0

# 재생할 파일의 넓이와 높이 읽어 두기
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = cap.get(cv2.CAP_PROP_FPS)


save_data = "./data"
if not os.path.exists(save_data):
    os.mkdir(save_data)

name = input("Insert Video Name(Only Alphabet):")
id = input("Insert Id(Non-Duplicate number):")

data_file = f"{id}_{name}.xlsx"

col = ['id','name','frame_idx','angle','frame_time']
df = pd.DataFrame(columns=col)

# dir = os.path.join(base_dir, name+'_'+ id)
# if not os.path.exists(dir):
#     os.mkdir(dir)


# 출력 영상 다시 저장할 객체 생성
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(save_result, fourcc, fps, (int(width), int(height)),True)
print("재생할 파일 넓이, 높이 : %d, %d"%(width, height))

if not cap.isOpened:
    print('--(!)Error opening video capture')
    exit(0)


with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while (cap.isOpened()):
        start_time = time.time()
        idx += 1
        success, frame = cap.read()    

        if frame is None:                                   # 동영상의 frame이 없으면 분석을 끝냅니다..
            # close the video file pointers
            cap.release()
            # close the writer point
            writer.release()
            print('--(!) No captured frame -- Break!')
            print("elapsed time {:.3f} seconds".format(elapsed_time))
            break
        if success == False:
            break

        # Recolor image to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  #BGR 포맷에서 RGB 포맷으로 이미지를 변환합니다.
        frame.flags.writeable = False

        # Make detection poses
        results_poses = poses.process(frame)

        # if results_hands.multi_hand_landmarks:                    # 읽어 드린 결과 중 손의 landmark 결과가 있는 경우 진행합니다.
        #     for handLMK in results_hands.multi_hand_landmarks:    # 읽어 드린 결과에서 찾은 손의 수(사용자 지정 기본 2)만큼 순환 진행
        #         for id, lmk in enumerate(handLMK.landmark): # 읽어 드린 손의 결과의 landmark 수만큼 순환 진행
        #             # print(id, lmk)
        #             h, w, c = frame.shape                                     # 원본 이미지의 크기
        #             cx, cy = int(lmk.x*w), int(lmk.y*h)                     # 해당 landmark의 좌표값을 곱해서 표시할 위치 확인
        #             # print(id, cx, cy)                                       # 해당 landmark의 위치 표시
        #             if id==0:                                               # 표시하고 싶은 손의 landmark 위치 지정 현재는 손바닥 중심(0번)
        #                 cv2.circle(frame,(cx,cy),15,(255,0,255),cv2.FILLED )  # 해당 위치에 원 표시
        #         mp_Draw.draw_landmarks(frame, handLMK, mpHands.HAND_CONNECTIONS
        #                               ,mp_drawing_styles.get_default_hand_landmarks_style()
        #                               ,mp_drawing_styles.get_default_hand_connections_style()) # mediapipe  draw 클래스로 landmark 위치점들 연결선 그리기

        if results_poses.pose_landmarks.landmark:
            landmarks = results_poses.pose_landmarks.landmark
            # Get coordinates
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            
            # Calculate angle
            angle = calculate_angle(shoulder,elbow, wrist)

            # Visualize angle
            cv2.putText(frame, str(angle), 
                            tuple(np.multiply(elbow, [640, 480]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )
        else:
            pass


        # Render detections
        mp_Draw.draw_landmarks(frame, results_poses.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_Draw.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                mp_Draw.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                    )
        
        # frame = cv2.resize(frame, (900, 600))         


        # Recolor back to BGR
        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


        elapsed_time = time.time()
        fps = 1/(elapsed_time-count_time)                                      # 프레임 처리 시간 계산
        count_time = elapsed_time

        # frame = cv2.flip(frame,1)
        cv2.putText(frame,str(int(fps)),(20,140),cv2.FONT_HERSHEY_PLAIN        # 프레임 처리시간 표시
                    , 3, (255,0,255),3)

        process_time = time.time() - start_time
        elapsed_time += process_time   # 총 경과시간 누적
        print("=== A idx {:d} frame took {:.3f} seconds".format(idx,process_time))

        data = [[id,name,idx,angle,process_time]]
        frame_data = pd.DataFrame(data,columns=col)

        # df.stack(frame_data,1)
        df = pd.concat([df,frame_data],axis=0)

        # frame의 저장
        if save_result is not None:
            writer.write(frame)

        cv2.imshow('frame', frame)
        if cv2.waitKey(10) == 27:
            break

    df.to_excel(os.path.join(save_data,data_file), header=True, index=False)

    cap.release()
    writer.release()
    cv2.destroyAllWindows()