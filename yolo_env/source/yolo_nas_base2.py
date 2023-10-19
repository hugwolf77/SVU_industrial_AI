from super_gradients.common.object_names import Models
from super_gradients.training import models
import random
import cv2
import numpy as np
import datetime
import torch


# GPU 설정
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
model = models.get(Models.YOLO_NAS_L, pretrained_weights="coco").to(device)

# global label_colors
# global names

# video 설정
video_path = 0
cap = cv2.VideoCapture(video_path)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# 코덱 및 비디오 쓰기 설정
# output_path = "/content/output_base.mp4"
# fourcc = cv2.VideoWriter_fourcc(*'MP4V')
# writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

while cap.isOpened() :
    # Start time to compute the FPS
    start = datetime.datetime.now()
    ret, frame = cap.read()
    if ret == True :
        first_frame = True
        # frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        outputs = model.predict(frame, conf=0.4, iou=0.4) # 
        output = outputs[0]
        bboxes = output.prediction.bboxes_xyxy
        confs = output.prediction.confidence
        labels = output.prediction.labels
        class_names = output.class_names

        if first_frame :
            random.seed(0)
            labels = [int(l) for l in list(labels)]
            label_colors = [tuple(random.choices(np.arange(0, 256), k=3)) 
            				for i in range(len(class_names))]
            names = [class_names[int(label)] for label in labels]
            first_frame = False

        for idx, bbox in enumerate(bboxes):
            bbox_left = int(bbox[0])
            bbox_top = int(bbox[1])
            bbox_right = int(bbox[2])
            bbox_bot = int(bbox[3])

            text = f"{names[idx]} {confs[idx]:.2f}"
            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, 1, 1)
            text_w, text_h = text_size
            colors = tuple(int(i) for i in label_colors[labels[idx]])
            cv2.rectangle(frame, (bbox_left, bbox_top),
            (bbox_left + text_w, bbox_top - text_h), colors, -1)
            cv2.putText(frame, text, (bbox_left, bbox_top),
            cv2.FONT_HERSHEY_PLAIN,1, (255, 255, 255), 1)
            cv2.rectangle(frame, (bbox_left, bbox_top), (bbox_right, bbox_bot), 
						color=colors, thickness=2) 

    # End time to compute the FPS
    end = datetime.datetime.now()
    fps = f"FPS: {1 / (end - start).total_seconds():.2f}"
    cv2.putText(frame, fps, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 8)
    # writer.write(frame)
    cv2.imshow('frame', frame)
    
    # Check for 'q' key press to exit the loop
    if cv2.waitKey(1) == ord("q"):
        break

# Release video capture and video writer objects
cap.release()
# writer.release()
# Close all windows
cv2.destroyAllWindows()