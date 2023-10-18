---
categories: Technology
title: YOLOv8 YOLO-NAS 소개
created: 2023-10-17 13:29
tags:
  - Vision
  - YOLO
---

## [[Yolo-NAS]]

### 1️⃣*NAS:Neural Archtecture Search*
1. *탐색공간(Search Space)* 검색 공간 또는 검색 공간으로 선택할 수 있는 유효한 아키텍처의 집합을 정의한다.
	- 탐색을 수행하는 공간
	- candidate operation (convolution, fully-connected, pooling, etc)
	- define count about operation
	- 각각이 하나의 유효한 네트워크와 이들에 의해 구성된 유효한 네트워크 구성의 재료 (valid network configuration)
2. *탐색전략(Search Strategy)* 검색 알고리즘으로 검색 공간에서 가능한 아키텍처를 전송하는 방법을 담당하는 검색 알고리즘
	-  search space에서 생성된 유효한 네트워크 구성 중 목표에 가장 합당한 구성을 찾는 알고리즘
	- random search, reinforcement learning, evolutionary strategy, gradient descent, bayesian optimization 등
	- exploration(탐색공간 커버)과 exploitation(효율적인 탐색) 모두를 수행해야 함.
	- exploration-exploitation trade-off
3. *성능평가전략(Performance Estimation Strategy)* 평가 전략으로, 후보 아키텍처를 비교하는 데 사용되는 가치 평가 전략
	-  후보 configuration에서 성능을 예측하고 평가, 더 좋은 configuration 추출할지 , 탐색을 지속할지 등을 결정 - 모든 경우를 시행하는 경우와 일부만으로 평가할 수 있는지 등 효율을 고려.
##### *NAS Examples*
1. NAS with Reinforcement Learning
2. AmoebaNet
3. Differentiable Architecture Search (DARTS)

### 2️⃣*YOLO-NAS 구성*
1) *Backbone* : Dense + Sparse, image feature extract part 
2) *Neck*: Backbone에서 추출된 특징을 향상, 다양한 스켈일에서 예측 생성
4) *Head*: 다중 스케일 피라미드를 사용 백본의  다양한 레벨에서의 특징을 결합 예측생성 (분류 분기 + 회귀 분기

>[! 양자화 (Quantization)] 
>모델 가중치의 정밀도를 낮춰 메모리 사용량을 줄이고 실행 속도를 높이는 과정
>- YOLO-NAS 양자화 기법은 int8  가중치 Float32 $\longrightarrow$ 1byte 메모리 절약 "EurVgg"사용해 정확도 손실을 줄임. 
>- Uranus Hybrid Quantization : 특정 레이어에만 양자화 적용 정보 소실과 latency  균형

![[YOLO-NAS.png|600]]
### 3️⃣*YOLO8과의 비교 평가*

>[!warning]
>- 정확한 비교라기 보다 몇몇의 의견
>- 🙂개인적 의견 : 제한 조건이 없는 상태에서 성능은 YOLOv8이 뛰어남 그러나 제한된 조건에서 사용하거나 커스터마이징 하는데에는 YOLO-NAS도 강점이 있음.
### [Ultralytics](https://github.com/ultralytics/ultralytics)
### [Ultralytics/yolo-nas](https://docs.ultralytics.com/models/yolo-nas/?h=nas#inference-and-validation-examples)
### [Deci-super-gradients](https://github.com/Deci-AI/super-gradients/blob/master/YOLONAS.md)

|영역|YOLO-NAS|YOLOv8|
|-------|-------|-------|
|작은 물체 감지|뛰어남|제한적|
|로컬라이제이션 정확성|뛰어남|제한적|
|학습 후 정량화|뛰어남|제한적|
|학습 후 정량화|용이함|어려움|
|online edge device app|가능함|가능함|
|정확도|높음|높음|
|처리속도|빠름|빠름|
|효율성|높음|높음|


### 3️⃣*YOLO-NAS의 사전 학습 데이터*
#### *pretrian dataset*
[Object365 bench mark dataset](https://www.objects365.org/overview.html)
#### *test dataset*
Roboflow A100 데이터셋에서 검출능력 확인

### 4️⃣*YOLO  Local 환경  테스트*

> [!NOTE] 객체 추적과 관련한 개념
> - SORT(Simple Online and Realtime Tracking) - real time object tracking 알고리즘을 사용하여 객체 추적이 이뤄짐
> - 객체식별은 YOLO, 객체추적은 SORT로 이뤄짐
> -  SORT 는 이전 Frame에서 인식된 객체(Object)와 다음 Frame 에서 인식된 객체가 같은 객체인지 평가하여 확인 (kallman filter 등 사용)

- 운영체제는 Window 입니다.
- 😀저의 경우 ==python3.8==  사용하였습니다. (😅 최신 python은 3.12 입니다. 기존 사용하던 환경 세팅이 귀찮아서 아직 Upgrade 하지 않고 있습니다.)
- 특정 디렉토리에 실습용 폴더📂를 만드세요. 저장 공간이 충분해야 합니다. 
- win-key + R 명령을 누르면  실행 입력창이 만들어 집니다. " cmd " 입력하면  cli 명령어를 입력할 수 있어요.
- "  CD [경로 또는 디렉토리명]" 을 입력하여 실습용 폴더로 이동하세요.

- Local 환경 cmd 명령 : python 은 설치되어 있는 것으로 봄
```bash or cmd
	python --version
	pip install virtualenv
	virtualenv env

	cd env/Scripts
	activate
	cd ..
	mkdir source
	cd source
```
- 보통은 requirement.txt 파일을 통해서 한번에 필요 package를 설치하지만 이번에는 하나씩 설치해 봅니다.
```bash or cmd
	# python 라이브러리 관리 프로그램 pip를 업그레이드 합니다.
	python.exe -m pip install --upgrade pip
	# 딥러닝에 많이 사용되는 torch 플랫폼(여기서는 cpu 버전을 설치합니다. GPU 가 있는 경우 cuda 사용 torcy를 설치하세요.)
	pip install torch torchvision torchaudio     
	# openCV2 이미지와 화상을 처리하는 라이브러리를 설치합니다.
	pip install opencv-python==4.6.0.66
	pip install opencv-contrib-python==4.6.0.66
	# YOLOv8 라이브러리를 설치합니다.
	pip install ultralytics
	# SORT 알고리즘인 deep-sort 알고리즘을 설치합니다.
	pip install deep-sort-realtime
	# YOLO-NAS 라이브러리를 설치합니다.
	pip install super-gradients
```


- Fine tunning 및 커스텀 traing에 참고되는 Reference 예제 너TUBE
https://youtu.be/PBh9MFH2lB4?si=F0xY48Dp_xnDR2IP
https://youtu.be/st9o5XqqNno?si=vqZRaG2QpA5FmEUV

