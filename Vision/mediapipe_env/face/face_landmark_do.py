'''
2024.10.30
    openCV and mediapipe landmark detect
    https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/tasks/python/vision/face_detector.py
    https://groups.google.com/g/mediapipe/c/uf5YDXuDbeQ?pli=1
'''

from types import NoneType
import numpy as np
import cv2
# import matplotlib.pyplot as plt
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

model_path = './pre/face_landmarker_v2_with_blendshapes.task'

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = vision.FaceLandmarker
FaceLandmarkerOptions = vision.FaceLandmarkerOptions
FaceLandmarkerResult = vision.FaceLandmarkerResult
VisionRunningMode = vision.RunningMode

cap = cv2.VideoCapture(0)

def draw_landmarks_on_image(rgb_image, detection_result):
  if type(detection_result) != NoneType:
     face_landmarks_list = detection_result.face_landmarks

  else:
     annotated_image = np.copy(rgb_image)
     return annotated_image
  
  annotated_image = np.copy(rgb_image)

  # Loop through the detected faces to visualize.
  for idx in range(len(face_landmarks_list)):
    face_landmarks = face_landmarks_list[idx]
    print(f"{idx}")

    # Draw the face landmarks.
    face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    face_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_tesselation_style())
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_contours_style())
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_IRISES,
          landmark_drawing_spec=None,
          connection_drawing_spec=mp.solutions.drawing_styles
          .get_default_face_mesh_iris_connections_style())
  return annotated_image


# Create a face landmarker instance with the live stream mode:
def print_result(result, output_image: mp.Image, timestamp_ms: int):
    # print('face landmarker result: {}'.format(result))
    annotated_image =draw_landmarks_on_image(output_image, result)
    return cv2.imshow('frame', cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)) # cv2.imshow('frame',annotated_image)

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)


with FaceLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False # Saves memory by making image not writeable    
        frame_timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        numpy_frame_from_opencv = np.array(image)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy_frame_from_opencv)
        # print(f"mp_image : {mp_image}")
        # detection_result = landmarker.detect_async(mp_image, frame_timestamp_ms)
        # annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), detection_result)
        
        landmarker.detect_async(mp_image, frame_timestamp_ms)

        # cv2.imshow('frame', cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
        # cv2.imshow('frame',annotated_image)
        if cv2.waitKey(10) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()