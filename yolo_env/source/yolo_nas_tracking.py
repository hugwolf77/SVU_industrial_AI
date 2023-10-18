#-- 필요 라이브러리 import
import time
import torch
import cv2
import torch.backends.cudnn as cudnn
from PIL import Image
import colorsys
import numpy as np
import math

from super_gradients.training import models
from super_gradients.common.object_names import Models

from .deep_sort.utils.parser import get_config
from .deep_sort.deep_sort import DeepSort
from .deep_sort.sort.tracker import Tracker

#-- GPU 설정
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

#-- 모델 설정
model = models.get("yolo_nas_s", pretrained_weights="coco").to(device)
conf_treshold = 0.70

#-- deep sort 알고리즘 설정
deep_sort_weights = "deep_sort/deep/checkpoint/ckpt.t7"
#-- max_age는 최대 몇 프레임까지 인정할지
tracker = DeepSort(model_path=deep_sort_weights, max_age=70)

#-- video 설정
video_path = "people.mp4"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error video file")
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

#-- 코덱 및 비디오 쓰기 설정
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_path = "output.mp4"
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))
fuse_model=False

frames = []
i = 0
count, fps, elapsed = 0, 0, 0
start_time = time.perf_counter()

coco128 = open('./yolov8_pretrained/coco128.txt', 'r')
classNames = coco128.read()
class_list = classNames.split('\n')
coco128.close()

totalCountUp = [] 
totalCountDown = [] 
limitup = [103, 161, 296, 161] 
limitdown = [527, 489, 735, 489]

while True:
    ret, frame = cap.read()  # 비디오 프레임 읽기
    count += 1  # 프레임 카운트 증가

    if ret:
        detections = np.empty((0, 5))

        # 모델을 사용하여 객체 검출 및 추적 수행
        result = list(model.predict(frame, conf=0.35))[0]
        bbox_xyxys = result.prediction.bboxes_xyxy.tolist()  # 객체의 경계상자 좌표
        confidences = result.prediction.confidence  # 객체의 신뢰도
        labels = result.prediction.labels.tolist()  # 객체의 레이블

        for (bbox_xyxy, confidence, cls) in zip(bbox_xyxys, confidences, labels):
            bbox = np.array(bbox_xyxy)
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            classname = int(cls)
            class_name = classNames[classname]
            conf = math.ceil((confidence*100))/100

            if class_name == "person" and conf > 0.3:
                currentarray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentarray))

        resultsTracker = tracker.update(detections)  # 객체 추적 업데이트

        # 경계선 그리기
        cv2.line(frame, (limitup[0], limitup[1]), (limitup[2], limitup[3]), (255,0,0), 3)  # 상한선
        cv2.line(frame, (limitdown[0], limitdown[1]), (limitdown[2], limitdown[3]), (255,0,0), 3)  # 하한선

        for result in resultsTracker:
            x1, y1, x2, y2, id = result
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # 객체를 사각형으로 표시
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 144, 30), 3)

            cx, cy = int((x1+x2)/2), int((y1+y2)/2)
            cv2.circle(frame, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

            label = f'{int(id)}'
            t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
            c2 = x1 + t_size[0], y1 - t_size[1] - 3

            # 객체 ID와 함께 사각형 위에 텍스트 표시
            cv2.rectangle(frame, (x1, y1), c2, [255, 0, 255], -1, cv2.LINE_AA)
            cv2.putText(frame, label, (x1, y1-2), 0, 1, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)

            # 상한선과 하한선을 통과한 객체 수 계산 및 표시
            if limitup[0] < cx < limitup[2] and limitup[1] - 15 < cy < limitup[3] + 15:
                if totalCountUp.count(id) == 0:
                    totalCountUp.append(id)
                    cv2.line(frame, (limitup[0], limitup[1]), (limitup[2], limitup[3]), (0, 255, 0), 3)

            if limitdown[0] < cx < limitdown[2] and limitdown[1] - 15 < cy < limitdown[3] + 15:
                if totalCountDown.count(id) == 0:
                    totalCountDown.append(id)
                    cv2.line(frame, (limitdown[0], limitdown[1]), (limitdown[2], limitdown[3]), (0, 255, 0), 3)

        # 상단 영역에 인원 수 표시
        cv2.rectangle(frame, (100, 65), (441, 97), [255, 0, 255], -1, cv2.LINE_AA)
        cv2.putText(frame, str("Person Entering") + ":" + str(len(totalCountUp)), (141, 91), 0, 1, [255, 255, 255], thickness=2, lineType=cv2.LINE_AA)

        # 하단 영역에 인원 수 표시
        cv2.rectangle(frame, (710, 65), (1100, 97), [255, 0, 255], -1, cv2.LINE_AA)
        cv2.putText(frame, str("Person Leaving") + ":" + str(len(totalCountDown)), (741, 91), 0, 1, [255, 255, 255], thickness=2, lineType=cv2.LINE_AA)

        resize_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        out.write(frame)

        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # 'q' 키를 누르면 반복문 종료
            break
    else:
        break
    
cap.release()
out.release()
cv2.destroyAllWindows()