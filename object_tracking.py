# 1. 필요한 라이브러리 호출 (5가지)
import numpy as np
import cv2
import pyautogui as auto
import time, datetime

# 2. CCTV 기본설정 (동영상 객체 생성, 객체의 가로/세로 설정)
cap = cv2.VideoCapture(r"G:\Coding_Alchemist\highway.mp4")               # pc에 탑재된 웹캠(동영상)의 객체 생성
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280) # 동영상 객체의 폭(pixcel기준)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720) # 동영상 객체의 높이(pixcel기준)
fps = cap.get(cv2.CAP_PROP_FPS)         # 동영상 객체의 초당 프레임수 읽기
delay = int(1000/fps)                   # 키보드 입력 대기 시간 설정 : 단위 밀리세컨드

# 3. 객체 추적을 위한 기본설정
in_tracking = False   # 객체 추적 여부(상태) 표시
roi = None            # 선택한 객체(물체)를 저장
tracker = cv2.TrackerCSRT_create()

# 4. CCTV 구동
while cap.isOpened():                   
    ret, frame = cap.read()             # 동영상 객체를 읽어서 frame 변수에 저장
    frame = cv2.flip(frame, 1)          # 동영상의 좌/우를 반전 : -1를 입력하면 상하반전
    key = cv2.waitKey(delay)            # 키보드 입력 대기 : delay변수만큼 대기
    now = datetime.datetime.now()       # 현재의 날짜/시각을 읽어 옴. 
    now = now.strftime("%Y/%m/%d %H:%M:%S") # 날짜/시각을 "연/월/일/시간:분:초"형태로 변형
    cv2.putText(frame, now, (10, 20),   # frame 객체에 날짜/시각을 넣기
                cv2.FONT_HERSHEY_TRIPLEX|cv2.FONT_ITALIC, # 문자의 폰트
                0.7, (0, 255, 0), 2)                      # 폰트 사이즈, 색상(B,G,R), 굵기
    if ret == False:            
        response = auto.alert(title = "경고",   # 경고창
                              text = "더 이상 동영상이 없습니다.") 
        if response == "OK": # 경고창의 "확인" 클릭
            break
    else:                     
        if key == ord("q"):  # 키보드 입력이 q라면
            response = auto.confirm(title = "확인",                  # 확인창 띄우기
                                    text = "CCTV를 종료하시겠습니까?")
            if response == "OK": # 확인창에서 "확인"버튼 클릭
                break            # 동영상 정지
            else:                # 확인창에서 "취소"버튼 클릭
                pass             # 아무 액션없이 지나감
        
        # 4-1. 객체 추적을 위한 관심영역(ROI) 설정과 tracker 초기화
        elif key == ord("t"):
            roi = cv2.selectROI("CCTV", frame, False)    # 관심영역(물체) 선택
            if roi is not None:
                tracker.init(frame, roi)                 # 추적 알고리즘 객체 초기화
                in_tracking = True                       # 객체 추적을 "실행(On)"으로 전환
            else:
                pass
            
        if in_tracking == True:
            _, (x, y, w, h) = tracker.update(frame)      # 실시간으로 프레임에서 ROI의 위치 업데이트
            if w and h:                                  # ROI위치 정보에서 변화(움직임)이 있다면,
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2) # 관심영역(물체)에 사각형 형성
            else:
                pass
        
        cv2.imshow("CCTV", frame)# 카메라에서 읽어온 동영상(이미지)를 "CCTV"창에 보여줌

# 위의 무한루프가 종료되면, 카메라에서 읽어온 동영상 객체 삭제 / 화면의 모든 창을 닫음
cap.release() 
cv2.destroyAllWindows()
# [출처] [파이썬 응용] OpenCV 컴퓨터 비젼 분야의 백미, 객체추적을 단 10분만에 간단하게 구현하기|작성자 코딩 연금술사