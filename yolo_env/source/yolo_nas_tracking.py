# 라이브러리 import
import datetime
import torch
import cv2
import torch.backends.cudnn as cudnn
from PIL import Image
import colorsys
import numpy as np
import math

from super_gradients.training import models
from super_gradients.common.object_names import Models
from deep_sort_realtime.deepsort_tracker import DeepSort

# GPU 설정
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
# 모델 설정
model = models.get("yolo_nas_l", pretrained_weights="coco").to(device)
conf_treshold = 0.70
# tracker 설정 : max_age는 최대 몇 프레임까지 인정할지
tracker = DeepSort(max_age=50)

# video 설정
video_path = "people.mp4"
cap = cv2.VideoCapture(video_path)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# 코덱 및 비디오 쓰기 설정
output_path = "output.mp4"
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
fuse_model=False

# coco classname 정보 => Yaml 로 교체
coco128 = open('./yolov8_pretrained/coco128.txt', 'r')
classNames = coco128.read()
class_list = classNames.split('\n')
coco128.close()

class_id = 0  # 사람만 지정
iou=0.5
conf= 0.5

# Create a list of random colors to represent each class
np.random.seed(42)  # to get the same colors
colors = np.random.randint(0, 255, size=(len(class_list), 3))  # (80, 3)

while True:
    # Start time to compute the FPS
    start = datetime.datetime.now()
    ret, frame = cap.read()  # 비디오 프레임 읽기
    if not ret:
        print('Cam Error')
        break
    detect = next(iter(model.predict(frame, iou=iou, conf=conf)))
    # Extract the bounding box coordinates, confidence scores, and class labels from the detection results
    bboxes_xyxy = torch.from_numpy(detect.prediction.bboxes_xyxy).tolist()
    confidence = torch.from_numpy(detect.prediction.confidence).tolist()
    labels = torch.from_numpy(detect.prediction.labels).tolist()
    # Combine the bounding box coordinates and confidence scores into a single list
    concate = [sublist + [element] for sublist, element in zip(bboxes_xyxy, confidence)]
    # Combine the concatenated list with the class labels into a final prediction list
    final_prediction = [sublist + [element] for sublist, element in zip(concate, labels)]
    # 결과 저장할 리스트 초기화
    results = []
    # Loop over the detections
    for data in final_prediction:
        # Extract the confidence (i.e., probability) associated with the detection
        confidence = data[4]
        # Filter out weak detections by ensuring the confidence is greater than the minimum confidence and with the class_id
        if class_id == None:
            if float(confidence) < conf:
                continue
        else:
            if ((int(data[5] != class_id)) or (float(confidence) < conf)):
                continue
        # If the confidence is greater than the minimum confidence, draw the bounding box on the frame
        xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
        class_id = int(data[5])
        # Add the bounding box (x, y, w, h), confidence, and class ID to the results list
        results.append([[xmin, ymin, xmax - xmin, ymax - ymin], confidence, class_id])
    # Update the tracker with the new detections
    tracks = tracker.update_tracks(results, frame=frame)
    # Loop over the tracks
    for track in tracks:
        # If the track is not confirmed, ignore it
        if not track.is_confirmed():
            continue
        # Get the track ID and the bounding box
        track_id = track.track_id
        ltrb = track.to_ltrb()
        class_id = track.get_det_class()
        x1, y1, x2, y2 = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])
        # Get the color for the class
        color = colors[class_id]
        B, G, R = int(color[0]), int(color[1]), int(color[2])
        # Create text for track ID and class name
        text = str(track_id) + " - " + str(class_list[class_id])
        # Draw bounding box and text on the frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (B, G, R), 2)
        cv2.rectangle(frame, (x1 - 1, y1 - 20), (x1 + len(text) * 12, y1), (B, G, R), -1)
        cv2.putText(frame, text, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    # End time to compute the FPS
    end = datetime.datetime.now()
    # Show the time it took to process 1 frame
    print(f"Time to process 1 frame: {(end - start).total_seconds() * 1000:.0f} milliseconds")
    # Calculate the frames per second and draw it on the frame
    fps = f"FPS: {1 / (end - start).total_seconds():.2f}"
    cv2.putText(frame, fps, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 8)

    # Show the frame
    cv2.imshow("Frame", frame)
    # Write the frame to the output video file
    writer.write(frame)
    # Check for 'q' key press to exit the loop
    if cv2.waitKey(1) == ord("q"):
        break

# Release video capture and video writer objects
cap.release()
writer.release()

# Close all windows
cv2.destroyAllWindows()
