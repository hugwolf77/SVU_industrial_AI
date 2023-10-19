import cv2
import numpy as np
import torch

from super_gradients.training import models
from super_gradients.common.object_names import Models

# GPU 설정
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
# 모델 설정
# Models.YOLOX_N , "yolo_nas_s"
model = models.get("yolo_nas_s", pretrained_weights="coco").to(device)
model.predict_webcam()