import numpy as np
import cv2
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import matplotlib.pyplot as plt


def draw_landmarks_on_image(rgb_image, detection_result):
  face_landmarks_list = detection_result.face_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected faces to visualize.
  for idx in range(len(face_landmarks_list)):
    face_landmarks = face_landmarks_list[idx]

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

def plot_face_blendshapes_bar_graph(face_blendshapes):
  # Extract the face blendshapes category names and scores.
  face_blendshapes_names = [face_blendshapes_category.category_name for face_blendshapes_category in face_blendshapes]
  face_blendshapes_scores = [face_blendshapes_category.score for face_blendshapes_category in face_blendshapes]
  # The blendshapes are ordered in decreasing score value.
  face_blendshapes_ranks = range(len(face_blendshapes_names))

  fig, ax = plt.subplots(figsize=(12, 12))
  bar = ax.barh(face_blendshapes_ranks, face_blendshapes_scores, label=[str(x) for x in face_blendshapes_ranks])
  ax.set_yticks(face_blendshapes_ranks, face_blendshapes_names)
  ax.invert_yaxis()

  # Label each bar with values
  for score, patch in zip(face_blendshapes_scores, bar.patches):
    plt.text(patch.get_x() + patch.get_width(), patch.get_y(), f"{score:.4f}", va="top")

  ax.set_xlabel('Score')
  ax.set_title("Face Blendshapes")
  plt.tight_layout()
  plt.show()

#--------------------------------------------------------------------------------------------------------

base_options = python.BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task')
options = vision.FaceLandmarkerOptions(base_options=base_options,
                                       output_face_blendshapes=True,
                                       output_facial_transformation_matrixes=True,
                                       num_faces=1)

detector = vision.FaceLandmarker.create_from_options(options)



# STEP 3: Load the input image.
image = mp.Image.create_from_file("image.png")

# STEP 4: Detect face landmarks from the input image.
detection_result = detector.detect(image)

# STEP 5: Process the detection result. In this case, visualize it.
annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
cv2_imshow(cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

#--------------------------------------------------------------------------------------------------------


import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a face landmarker instance with the live stream mode:
def print_result(result: FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    print('face landmarker result: {}'.format(result))

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)

with FaceLandmarker.create_from_options(options) as landmarker:
  # The landmarker is initialized. Use it here.
  # ...

  # Use OpenCV’s VideoCapture to start capturing from the webcam.

  # Create a loop to read the latest frame from the camera using VideoCapture#read()

  # Convert the frame received from OpenCV to a MediaPipe’s Image object.
  mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy_frame_from_opencv)
  # Send live image data to perform face landmarking.
  # The results are accessible via the `result_callback` provided in
  # the `FaceLandmarkerOptions` object.
  # The face landmarker must be created with the live stream mode.
  landmarker.detect_async(mp_image, frame_timestamp_ms)
    


# 웹캠, 영상 파일의 경우 이것을 사용하세요.:

cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)

'''
gpu_src = cv2.cuda_GpuMat()
gpu_src.upload(cap)

terminate called after throwing an instance of 'cv::Exception'
  what():  OpenCV(4.8.1) /io/opencv/modules/core/include/opencv2/core/private.cuda.hpp:106: error: (-216:No CUDA support) The library is compiled without CUDA support in function 'throw_no_cuda'
'''

# with mp_face_mesh.FaceMesh(
#         max_num_faces=1,
#         refine_landmarks=True,
#         min_detection_confidence=0.5,
#         min_tracking_confidence=0.5) as face_mesh:
#     while cap.isOpened():
#         success, image = cap.read()
#         if not success:
#             print("웹캠을 찾을 수 없습니다.")
#             # 비디오 파일의 경우 'continue'를 사용하시고, 웹캠에 경우에는 'break'를 사용하세요
#             continue

#         # 필요에 따라 성능 향상을 위해 이미지 작성을 불가능함으로 기본 설정합니다.
#         image.flags.writeable = False
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         results = face_mesh.process(image)

#         # 이미지 위에 얼굴 그물망 주석을 그립니다.
#         image.flags.writeable = True
#         image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#         if results.multi_face_landmarks:
#             for face_landmarks in results.multi_face_landmarks:
#                 mp_drawing.draw_landmarks(
#                     image=image,
#                     landmark_list=face_landmarks,
#                     connections=mp_face_mesh.FACEMESH_TESSELATION,
#                     landmark_drawing_spec=None,
#                     connection_drawing_spec=mp_drawing_styles
#                     .get_default_face_mesh_tesselation_style())
#                 mp_drawing.draw_landmarks(
#                     image=image,
#                     landmark_list=face_landmarks,
#                     connections=mp_face_mesh.FACEMESH_CONTOURS,
#                     landmark_drawing_spec=None,
#                     connection_drawing_spec=mp_drawing_styles
#                     .get_default_face_mesh_contours_style())
#                 mp_drawing.draw_landmarks(
#                     image=image,
#                     landmark_list=face_landmarks,
#                     connections=mp_face_mesh.FACEMESH_IRISES,
#                     landmark_drawing_spec=None,
#                     connection_drawing_spec=mp_drawing_styles
#                     .get_default_face_mesh_iris_connections_style())
#         # 보기 편하게 이미지를 좌우 반전합니다.
#         cv2.imshow('MediaPipe Face Mesh(Puleugo)', cv2.flip(image, 1))
#         if cv2.waitKey(5) & 0xFF == 27:
#             break
# cap.release()