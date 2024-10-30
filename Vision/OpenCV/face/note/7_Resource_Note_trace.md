목차
1. 비슷한 그림 찾기
2. 영상의 특징과 키 포인트
3. 디스크립터 추출기
4. 특징 매칭

1. 비슷한 그림 찾기
영상 속 객체를 인식하는 방법 중 하나가 비슷한 그림을 찾아내는 것입니다.

새로운 영상이 입력되면 이미 알고 있던 영상들 중에 가장 비슷한 영상을 찾아 그 영상에 있는 객체로 판단하는 것입니다.

1.1 평균 해시 매칭

평균 해시는 어떤 영상이든 동일한 크기의 하나의 숫자로 변환되는데, 이때 숫자를 얻기 위해 평균 값을 이용한다는 뜻입니다.

평균을 얻기 전에 영상을 가로 세로 비율과 무관하게 특정한 크기로 축소합니다. 그 다음, 픽셀 전체의 평균 값을 구해서 각 픽셀의 값이 평균보다 작으면 0, 크면 1로 바꿔줍니다.

```python
import cv2

#영상 읽어서 그레이 스케일로 변환
img = cv2.imread('./img/pistol.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 8x8 크기로 축소 ---①
gray = cv2.resize(gray, (16,16))
# 영상의 평균값 구하기 ---②
avg = gray.mean()
# 평균값을 기준으로 0과 1로 변환 ---③
bin = 1 * (gray > avg)
print(bin)

# 2진수 문자열을 16진수 문자열로 변환 ---④
dhash = []
for row in bin.tolist():
    s = ''.join([str(i) for i in row])
    dhash.append('%02x'%(int(s,2)))
dhash = ''.join(dhash)
print(dhash)

cv2.namedWindow('pistol', cv2.WINDOW_GUI_NORMAL)
cv2.imshow('pistol', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

해밍 거리는 두 수의 같은 자리의 값이 서루 다른 것이 몇 개인지 나타내는 것입니다.

이제 1만 장에 가까운 사진들 중에서 앞서 다룬 권총 이미지와 비슷한 이미지를 찾아봅시다. (해밍 거리 25% 이내만 출력)

```python
import cv2
import numpy as np
import glob

# 영상 읽기 및 표시
img = cv2.imread('./img/pistol.jpg')
cv2.imshow('query', img)

# 비교할 영상들이 있는 경로 ---①
search_dir = './img/101_ObjectCategories'

# 이미지를 16x16 크기의 평균 해쉬로 변환 ---②
def img2hash(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (16, 16))
    avg = gray.mean()
    bi = 1 * (gray > avg)
    return bi

# 해밍거리 측정 함수 ---③
def hamming_distance(a, b):
    a = a.reshape(1,-1)
    b = b.reshape(1,-1)
    # 같은 자리의 값이 서로 다른 것들의 합
    distance = (a !=b).sum()
    return distance

# 권총 영상의 해쉬 구하기 ---④
query_hash = img2hash(img)

# 이미지 데이타 셋 디렉토리의 모든 영상 파일 경로 ---⑤
img_path = glob.glob(search_dir+'/**/*.jpg')
for path in img_path:
    # 데이타 셋 영상 한개 읽어서 표시 ---⑥
    img = cv2.imread(path)
    cv2.imshow('searching...', img)
    cv2.waitKey(5)
    # 데이타 셋 영상 한개의 해시  ---⑦
    a_hash = img2hash(img)
    # 해밍 거리 산출 ---⑧
    dst = hamming_distance(query_hash, a_hash)
    if dst/256 < 0.25: # 해밍거리 25% 이내만 출력 ---⑨
        print(path, dst/256)
        cv2.imshow(path, img)
cv2.destroyWindow('searching...')
cv2.waitKey(0)
cv2.destroyAllWindows()
```

1.2 템플릿 매칭

템플릿 매칭은 어떤 물체가 있는 영상을 준비해두고 그 물체가 포함되어 있을 것이라고 예상할 수 있는 입력 영상과 비교해서 물체가 매칭되는 위치를 찾는 것입니다.

OpenCV는 템플릿 매칭 cv2.matchTemplate()함수를 제공합니다.

result = cv2.matchTemplate(img, templ, method[, result, mask])

img : 입력 영상
templ : 템플릿 영상
method : 매칭 메서드
cv2.TM_SQDIFF : 제곱 차이 매칭, 완벽 매칭: 0, 나쁜 매칭: 큰 값
cv2.TM_SQDIFF_NORMED : 제곱 차이 매칭의 정규화
cv2.TM_CCORR : 상관관계 매칭, 완벽 매칭: 큰 값, 나쁜 매칭: 0
cv2.TM_CCORR_NORMED : 상관관계 매칭의 정규화
cv2.TM_CCOEFF : 상관계수 매칭, 완벽 매칭: 1, 나쁜 매칭: -1
result : 매칭 결과 2차원 배열
mask : TM_SQDIFF, TM_CCORR_NORMED인 경우 사용할 마스크
minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc( src[, mask])

src : 입력 1채널 배열
minVal, maxVal : 배열 전체에서 최소값, 최대값
minLoc, maxLoc : 최소값과 최대값의 좌표(x, y)

```python
import cv2
import numpy as np

# 입력이미지와 템플릿 이미지 읽기
img = cv2.imread('./img/figures.jpg')
template = cv2.imread('./img/pica.jpg')
th, tw = template.shape[:2]
cv2.imshow('template', template)

# 3가지 매칭 메서드 순회
methods = ['cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF_NORMED']
for i, method_name in enumerate(methods):
    img_draw = img.copy()
    method = eval(method_name)
    # 템플릿 매칭   ---①
    res = cv2.matchTemplate(img, template, method)
    # 최대, 최소값과 그 좌표 구하기 ---②
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    print(method_name, min_val, max_val, min_loc, max_loc)

    # TM_SQDIFF의 경우 최소값이 좋은 매칭, 나머지는 그 반대 ---③
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
        match_val = min_val
    else:
        top_left = max_loc
        match_val = max_val
    # 매칭 좌표 구해서 사각형 표시   ---④      
    bottom_right = (top_left[0] + tw, top_left[1] + th)
    cv2.rectangle(img_draw, top_left, bottom_right, (0,0,255),2)
    # 매칭 포인트 표시 ---⑤
    cv2.putText(img_draw, str(match_val), top_left, cv2.FONT_HERSHEY_PLAIN, 2,(0,255,0), 1, cv2.LINE_AA)
    cv2.imshow(method_name, img_draw)
cv2.waitKey(0)
cv2.destroyAllWindows()    
```

템플릿 매칭은 크기, 방향, 회전 등의 변화에는 잘 검출되지 않고 속도가 느리다는 단점이 있습니다.

2. 영상의 특징과 키 포인트
지금까지 다룬 특징 추출과 매칭 방법은 영상 전체를 전역적으로 반영하는 방법입니다. 전역적 매칭은 비교하려는 두 영상의 내용이 거의 대부분 비슷해야 하며, 다른 물체에 가려지거나 회전이나 방향, 크기 변화가 있으면 효과가 없습니다.

따라서 영상에서 여러 개의 지역적 특징을 표현할 수 있는 방법이 필요합니다.

2.1 코너 특징 검출

사람은 영상 속 객체를 판단할 때 주로 픽셀의 변화가 심한 곳에 중점적으로 관심을 둡니다. 그 중에서도 엣지와 엣지가 만나는 코너(corner)에 가장 큰 관심을 두게 됩니다.

해리스 코너 검출(Harris corner detection)은 소벨(Sobel) 미분으로 엣지를 검출하면서 엣지의 경사도 변화량을 측정하여 변화량이 X축과 Y축 모든 방향으로 크게 변화하는 것을 코너로 판단합니다.


dst = cv2.cornerHarris(src, blockSize, ksize, k[, dst, borderType]
src : 입력 영상, 그레이 스케일
blockSize : 이웃 픽셀 범위
ksize : 소벨 미분 커널 크기
k : 코너 검출 상수, 경험적 상수(0.04~0.06)
dst : 코너 검출 결과
src와 같은 크기와 1채널 배열, 변화량의 값, 지역 최대 값이 코너점을 의미
borderType : 외곽 영역 보정 형식

```python
import cv2
import numpy as np

img = cv2.imread('./img/house.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 해리스 코너 검출 ---①
corner = cv2.cornerHarris(gray, 2, 3, 0.04)
# 변화량 결과의 최대값 10% 이상의 좌표 구하기 ---②
coord = np.where(corner > 0.1* corner.max())
coord = np.stack((coord[1], coord[0]), axis=-1)

# 코너 좌표에 동그리미 그리기 ---③
for x, y in coord:
    cv2.circle(img, (x,y), 5, (0,0,255), 1, cv2.LINE_AA)

# 변화량을 영상으로 표현하기 위해서 0~255로 정규화 ---④
corner_norm = cv2.normalize(corner, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
# 화면에 출력
corner_norm = cv2.cvtColor(corner_norm, cv2.COLOR_GRAY2BGR)
merged = np.hstack((corner_norm, img))
cv2.imshow('Harris Corner', merged)
cv2.waitKey()
cv2.destroyAllWindows()
```

변화량 결과 최대값 10%이상의 좌표에만 빨간색 동그라미로 표시하였습니다.

시(Shi)와 토마시(Tomasi)는 해리스 코너 검출을 개선한 알고리즘을 발표했습니다. 이 방법으로 검출한 코너는 객체 추적에 좋은 특징이 된다고 해서 OpenCV는 cv2.goodFeaturesToTrack()이라는 이름의 함수를 제공합니다.

corners = cv2.goodFeaturesToTrack(img, maxCorners, qualityLevel, minDistance[, corners, mask, blockSize, useHarrisDetector, k])
img : 입력 영상
maxCorners : 얻고 싶은 코너 개수, 강한 것 순
qualityLevel : 코너로 판단할 스레시홀드 값
minDistance : 코너 간 최소 거리
mask : 검출에 제외할 마스크
blockSize=3 : 코너 주변 영역의 크기
useHarrisDetector=False : 코너 검출 방법 선택
True = 해리스 코너 검출 방법, False = 시와 토마시 검출 방법
k : 해리스 코너 검출 방법에 사용할 k 계수
corners : 코너 검출 좌표 결과, N x 1 x 2 크기의 배열, 실수 값 (정수 변형 필요)

```python
import cv2
import numpy as np

img = cv2.imread('./img/house.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 시-토마스의 코너 검출 메서드
corners = cv2.goodFeaturesToTrack(gray, 80, 0.01, 10)
# 실수 좌표를 정수 좌표로 변환
corners = np.int32(corners)

# 좌표에 동그라미 표시
for corner in corners:
    x, y = corner[0]
    cv2.circle(img, (x, y), 5, (0,0,255), 1, cv2.LINE_AA)

cv2.imshow('Corners', img)
cv2.waitKey()
cv2.destroyAllWindows()
```

2.2 키 포인트와 특징 검출기

OpenCV는 모든 특징 검출기를 cv2.Feature2D 클래스를 상속받아 구현했으며, 이것으로부터 추출된 특징점은 cv2.KeyPoint라는 객체에 담아 표현합니다. cv2.Feature2D를 상속받아 구현된 특징 검출기는 모두 12가지이고, 그 중 6가지만 다뤄보도록 하겠습니다.


keypoints = detector.detect(img [, mask]) : 키 포인트 검출 함수

img : 입력 영상, 바이너리 스케일
mask : 검출 제외 마스크
keypoints : 특징점 검출 결과, KeyPoint의 리스트
cv2.Keypoint : 특징점 정보를 담는 객체

pt : 키 포인트(x, y) 좌표, float 타입 (정수 변환 필요)
size : 의미있는 키 포인트 이웃의 반지름
angle : 특징점 방향 (시계방향, -1 = 의미없음)
response : 특징점 반응 강도 (추출기에 따라 다름)
octave : 발견된 이미지 피라미드 계층
class_id : 키 포인트가 속한 객체 ID
키 포인트 속성 중에 특징점의 좌표 정보인 pt 속성은 항상 값을 갖지만 나머지 속성은 사용하는 검출기에 따라 채워지지 않는 경우도 있습니다.


검출한 키 포인트를 영상에 표시하고 싶을 때는 cv2.drawKeypoints() 함수를 이용하면 됩니다.

outImg = cv2.drawKeypoints(img, keypoints, outImg[, color[, flags]])
img : 입력 이미지
keypoints : 표시할 키 포인트 리스트
outImg : 키 포인트가 그려진 결과 이미지
color : 표시할 색상 (기본 값: 랜덤)
flags : 표시 방법 선택 플래그
cv2.DRAW_MATCHES_FLAGS_DEFAULT : 좌표 중심에 동그라미만 그림 (기본 값)
cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS : 동그라미의 크기를 size와 angle을 반영하여 그림
2.3 GFTTDetector

GFTTDetector는 cv2.goodFeaturesToTrack() 함수로 구현된 특징 검출기 입니다.

detoctor = cv2.GFTTDetector_create([, maxCorners[, qualityLevel, minDistance, blockSize, useHarrisDetector, k])
인자의 모든 내용은 cv2.goodFeaturesToTrack()과 동일

```python
import cv2
import numpy as np
 
img = cv2.imread("./img/house.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Good feature to trac 검출기 생성 ---①
gftt = cv2.GFTTDetector_create() 
# 키 포인트 검출 ---②
keypoints = gftt.detect(gray, None)
# 키 포인트 그리기 ---③
img_draw = cv2.drawKeypoints(img, keypoints, None)

# 결과 출력 ---④
cv2.imshow('GFTTDectector', img_draw)
cv2.waitKey()
cv2.destroyAllWindows()
```

2.4 FAST

FAST(Feature from Accelerated Segment Test)는 속도를 개선한 알고리즘입니다. 코너를 검출할 때 미분 연산으로 엣지 검출을 하지 않고 픽셀을 중심으로 특정 개수의 픽셀로 원을 그려서 그 안의 픽셀들이 중심 픽셀 값보다 임계 값 이상 밝거나 어두운 것이 특정 개수 이상 연속되면 코너로 판단합니다.

detector = cv2.FastFeatureDetector_create([threshold [, nonmaxSuppression, type])
threshold=10 : 코너 판단 임계 값
nonmaxSuppression = True : 최대 점수가 아닌 코너 억제
type : 엣지 검출 패턴
cv2.FastFeatureDetector_TYPE_9_16 : 16개 중 9개 연속 (기본 값)
cv2.FastFeatureDetector_TYPE_7_12 : 12개 중 7개 연속
cv2.FastFeatureDetector_TYPE_5_8 : 8개 중 5개 연속

```python
import cv2
import numpy as np

img = cv2.imread('./img/house.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# FASt 특징 검출기 생성 ---①
fast = cv2.FastFeatureDetector_create(50)
# 키 포인트 검출 ---②
keypoints = fast.detect(gray, None)
# 키 포인트 그리기 ---③
img = cv2.drawKeypoints(img, keypoints, None)
# 결과 출력 ---④
cv2.imshow('FAST', img)
cv2.waitKey()
cv2.destroyAllWindows()
```

2.5 SimpleBlobDetector

BLOB(Binary Large Object)는 바이너리 스케일 이미지의 연결된 픽셀 그룹을 말하는 것으로, 자잘한 객체는 노이즈로 판단하고 특정 크기 이상의 큰 객체에만 관심을 두는 방법입니다.

detector = cv2.SimpleBlobDetector_create( [parameters] ) : BLOB 검출기 생성자

paramters : BLOB 검출 필터 인자 객체
cv2.SimpleBlobDetector_Params()

minThreshold, maxThreshold, thresholdStep : BLOB를 생성하기 위한 경계 값 (minThreshold에서 maxThreshold를 넘지 않을 때까지 thresholdStep만큼 증가)
minRepeatability : BLOB에 참여하기 위한 연속된 경계 값의 개수
minDistBetweenBlobs : 두 BLOB를 하나의 BLOB로 간주한 거리
filterByArea : 면적 필터 옵션
minArea, maxArea : min~max 범위의 면적만 BLOB로 검출
filterByCircularity : 원형 비율 필터 옵션
minCircularity, maxCircularity : min~max 범위의 원형 비율만 BLOB로 검출
filterByColor : 밝기를 이용한 필터 옵션
blobColor : 0 = 검은색 BLOB 검출, 255 = 흰색 BLOB 검출
filterByConvexity : 볼록 비율 필터 옵션
minConvexity, maxConvexity : min~max 범위의 볼록 비율만 BLOB로 검출
filterByInertia : 관성 비율 필터 옵션
minIneriaRatio, maxInertiaRatio : min~max 범위의 관성 비율만 BLOB로 검출

```python
import cv2
import numpy as np
 
img = cv2.imread("./img/house.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# SimpleBlobDetector 생성 ---①
detector = cv2.SimpleBlobDetector_create()
# 키 포인트 검출 ---②
keypoints = detector.detect(gray)
# 키 포인트를 빨간색으로 표시 ---③
img = cv2.drawKeypoints(img, keypoints, None, (0,0,255),\
                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
 
cv2.imshow("Blob", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

BLOB로 검출된 창문 몇 개에 빨간색 동그라미가 그려진 것을 볼 수 있습니다.

3. 디스크립터 추출기
3.1 특징 디스크립터와 추출기

특징 디스크립터(feature descriptor)는 키 포인트 주변 픽셀을 일정한 크기의 블록으로 나눠어 각 블록에 속한 픽셀의 그레이디언트 히스토그램을 계산한 것으로, 키 포인트 주위의 밝기, 색상, 방향, 크기 등의 정보를 표현한 것입니다.

특징 디스크립터를 추출하기 위한 방법으로 cv2.Feature2D 클래스를 상속받아 구현하였습니다.

keypoints, descriptors = detector.compute(image, keypoints[, descriptors]) : 키 포인트를 전달하면 특징 디스크립터를 계산해서 반환

keypoints, descriptors = detector.detectAndCompute(image, mask[, descriptors, useProvidedKeypoints]) : 키 포인트 검출과 특징 디스크립터 계산을 한번에 수행

image : 입력 영상
keypoints : 디스크립터 계산을 위해 사용할 키 포인트
descriptors : 계산된 디스크립터
mask : 키 포인트 검출에 사용할 마스크
useProvidedKeypoints : True인 경우 키 포인트 검출을 수행하지 않음 (사용 안 함)
앞서 다룬 GFTTDetector와 SimpleBlobDetector는 compute()와 detectAndCompute()가 구현되어 있지 않습니다.

앞으로 다룰 여러 디스크립터 추출기는 키 포인트를 얻기 위한 detect() 함수와 키 포인트로 디스크립터를 얻기 위한 compute() 함수가 모두 구현되어 있어 각각의 함수를 사용하지 않고 detectAndCompute() 함수를 사용하는 것이 편리합니다.

3.2 SIFT

SIFT(Scale-Invariant Feature Transform)는 이미지 피라미드를 이용해서 크기 변화에 따른 특징 검출의 문제를 해결한 알고리즘입니다.

detector = cv2.xfeatures2d.SIFT_create([, nfeatures[, nOctaveLayers[, contrastThreshold[, edgeThreshold[, sigma]]]]])
nfeatures : 검출 최대 특징 수
nOctaveLayers : 이미지 피라미드에 사용할 계층 수
contrastThreshold : 필터링할 빈약한 특징 문턱 값
edgeThreshold : 필터링할 엣지 문턱 값
sigma : 이미지 피라미드 0 계층에서 사용할 가우시안 필터의 시그마 값

```python
import cv2
import numpy as np

img = cv2.imread('./img/house.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# SIFT 추출기 생성
sift = cv2.xfeatures2d.SIFT_create()
# 키 포인트 검출과 서술자 계산
keypoints, descriptor = sift.detectAndCompute(gray, None)
print('keypoint:',len(keypoints), 'descriptor:', descriptor.shape)
print(descriptor)

# 키 포인트 그리기
img_draw = cv2.drawKeypoints(img, keypoints, None, \
                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# 결과 출력
cv2.imshow('SIFT', img_draw)
cv2.waitKey()
cv2.destroyAllWindows()
```

3.3 ORB

ORB(Oriented and Rotated BRIEF)는 특징 검출을 지원하지 않는 디스크립터 추출기인 BRIEF에 방향과 회전을 고려하도록 개선한 알고리즘입니다.

dectector = cv.ORB_create([nfeatures, scaleFactor, nlevels, edgeThreshold, firstLevel, WTA_K, scoreType, patchSize, fastThreshold])
nfeatures = 500 : 검출할 최대 특징 수
scaleFactor = 1.2 : 이미지 피라미드 비율
nlevels = 8 : 이미지 피라미드 계층 수
edgeThreshold = 31 : 검색에서 제외할 테두리 크기, patchSize와 맞출 것
firstLevel = 0 : 최초 이미지 피라미드 계층 단계
WTA_K = 2 : 임의 좌표 생성 수
scoreType : 키 포인트 검출에 사용할 방식
cv2.ORB_HARRIS_SCORE : 해리스 코너 검출 (기본 값)
cv2.ORB_FAST_SCORE : FAST 코너 검출
patchSize = 31 : 디스크립터의 패치 크기
fastThreshold = 20 : FAST에 사용할 임계 값

```python
import cv2
import numpy as np

img = cv2.imread('./img/house.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ORB 추출기 생성
orb = cv2.ORB_create()
# 키 포인트 검출과 서술자 계산
keypoints, descriptor = orb.detectAndCompute(img, None)
# 키 포인트 그리기
img_draw = cv2.drawKeypoints(img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# 결과 출력
cv2.imshow('ORB', img_draw)
cv2.waitKey()
cv2.destroyAllWindows()
```

4. 특징 매칭
특징 매칭(feature matching)이란 서로 다른 두 영상에서 구한 키 포인트와 특징 디스크립터들을 각각 비교해서 그 거리가 비슷한 것끼리 짝짓는 것을 말합니다. 짝지어진 특징점들 중에 거리가 유의미한 것들을 모아서 대칭점으로 표시하면 그 개수에 따라 두 영상이 얼마나 비슷한지 측정할 수 있고 충분히 비슷한 영상이라면 비슷한 모양의 영역을 찾아낼 수도 있습니다. 특징 매칭은 파노라마 사진 생성, 이미지 검색, 등록한 객체 인식 등 다양하게 응용할 수 있습니다.

4.1 특징 매칭 인터페이스
matcher = cv2.DescriptorMatcher_create(matcherType) : 매칭기 생성자
matcherType : 생성할 구현 클래스의 알고리즘, 문자열
"BruteForce" : NORM_L2를 사용하는 BFMatcher
"BruteForce-L1" : NORM_L1을 사용하는 BFMatcher
"BruteForce-Hamming" : NORM_HAMMING을 사용하는 BFMatcher
"BruteForce-Hamming(2)" : NORM_HAMMING2를 사용하는 BFMatcher
"FlannBased" : NORM_L2를 사용하는 FlannBasedMatcher
matches = matcher.match(queryDescriptors, trainDescriptors[, mask]): 1개의 최적 매칭
queryDescriptors : 특징 디스크립터 배열, 매칭의 기준이 될 디스크립터
trainDescriptors : 특징 디스크립터 배열, 매칭의 대상이 될 디스크립터
mask : 매칭 진행 여부 마스크
matches : 매칭 결과
matches = matcher.knnMatch(queryDescriptors, trainDescriptors, k[, mask[, compactResult]]): k개의 가장 근접한 매칭
k : 매칭할 근접 이웃 개수
compactResult = False : 매칭이 없는 경우 매칭 결과에 불포함
matches = matcher.radiusMatch(queryDescriptors, trainDescriptors, maxDistance[, mask, compactResult]) : maxDistance 이내의 거리 매칭
maxDistance : 매칭 대상 거리
DMatch : 매칭 결과를 표현하는 객체
queryIdx : queryDescriptor의 인덱스
trainIdx : trainDescriptor의 인덱스
imgIdx : trainDescriptor의 이미지 인덱스
distance : 유사도 거리
cv2.drawMatches(img1, kp1, img2, kp2, matches, flags) : 매칭점을 영상에 표시
img1, kp1 : queryDescriptor의 영상과 키 포인트
img2, kp2 : trainDescriptor의 영상과 키 포인트
matches : 매칭 결과
flags : 매칭점 그리기 옵션
cv2.DRAW_MATCHES_FLAGS_DEFAULT : 결과 이미지 새로 생성
cv2.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG : 결과 이미지 새로 생성 안함
cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS : 키 포인트 크기와 방향도 그리기
cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS : 한쪽만 있는 매칭 결과 그리기 제외
OpenCV에서는 BFMatcher와 FLannBasedMatcher 특징 매칭기를 제공합니다. 매칭기 객체를 만들고 나면 두 개의 디스크립터를 이용해서 매칭해주는 3개의 함수 match, knnMatch, radiusMatch가 있습니다. 3개의 함수 모두 첫 번째 인자인 queryDescriptor를 기준으로 두 번째 인자인 trainDescriptor에 맞는 매칭을 찾습니다.

match함수는 queryDescriptor 1개당 최적 매칭을 이루는 trainDescriptor 1개를 찾아 결과에 반영합니다.
knnMatch함수는 queryDescriptor 1개당 k인자에 전달한 최근접 이웃 개수만큼 trainDescriptor에서 찾아 결과에 반영합니다.
radiusMatch() 함수는 queryDescriptor에서 maxDistance 이내에 있는 train Descriptor를 찾아 결과 매칭에 반영합니다.

4.2 BFMatcher
Brute-Force 매칭기는 일일이 전수조사를 하여 매칭을 하는 알고리즘 입니다.

matcher = cv.BFMatcher_create([normType[, crossCheck])
normType : 거리 측정 알고리즘 ( NORM_L1, NORM_L2, NORM_HAMMING 등 )
crossCheck = False : 상호 매칭이 있는 것만 반영
거리 측정 알고리즘은 3가지 유클리드 거리와 2가지 해밍 거리 중에 선택할 수 있습니다. SIFT, SURF로 추출한 디스크립터에 경우 L1, L2 방법이 적합하고 ORB는 HAMMING, ORB의 WTA_K가 3 or 4인 경우에는 HAMMING2가 적합합니다. crossCheck를 True로 설정하면 양쪽 디스크립터 모두에게서 매칭이 완성된 것만 반영하므로 불필요한 매칭을 줄일 수 있지만 그만큼 속도가 느려집니다.


```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

img1 = cv2.imread('./img/yungyung.png')
img2 = cv2.imread('./img/3yung.png')
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# SIFT 서술자 추출기 생성 ---①
detector = cv2.xfeatures2d.SIFT_create()
# 각 영상에 대해 키 포인트와 서술자 추출 ---②
kp1, desc1 = detector.detectAndCompute(gray1, None)
kp2, desc2 = detector.detectAndCompute(gray2, None)

# BFMatcher 생성, L1 거리, 상호 체크 ---③
matcher = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
# 매칭 계산 ---④
matches = matcher.match(desc1, desc2)
# 매칭 결과 그리기 ---⑤
res = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, \
                      flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
# 결과 출력 

plt.figure(figsize = (10,6))
imgs = {'BFMatcher + SIFT':res}
for i, (k, v) in enumerate(imgs.items()):
    plt.subplot(1,1,i+1)
    plt.title(k)
    plt.imshow(v[:,:,(2,1,0)])
    plt.xticks([]),plt.yticks([])

plt.show()
```

```python
# ORB

import cv2, numpy as np

img1 = cv2.imread('./img/yungyung.png')
img2 = cv2.imread('./img/3yung.png')
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# SIFT 서술자 추출기 생성 ---①
detector = cv2.ORB_create()
# 각 영상에 대해 키 포인트와 서술자 추출 ---②
kp1, desc1 = detector.detectAndCompute(gray1, None)
kp2, desc2 = detector.detectAndCompute(gray2, None)

# BFMatcher 생성, Hamming 거리, 상호 체크 ---③
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# 매칭 계산 ---④
matches = matcher.match(desc1, desc2)
# 매칭 결과 그리기 ---⑤
res = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, \
                     flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)


plt.figure(figsize = (10,6))
imgs = {'BFMatcher + ORB':res}
for i, (k, v) in enumerate(imgs.items()):
    plt.subplot(1,1,i+1)
    plt.title(k)
    plt.imshow(v[:,:,(2,1,0)])
    plt.xticks([]),plt.yticks([])

plt.show()
```

4.3 FLANN
BFMatcher는 특징 디스크립터를 전수 조사하기 때문에 사용하는 영상이 큰 경우에 속도가 느려진다는 단점이 있습니다. FLANN(Fast Library for Approximate Nearest Neighbors Matching)은 모든 특징 디스크립터를 비교하기 보다는 가장 가까운 이웃의 근사 값으로 매칭합니다.

matcher = cv2.FlannBasedMatcher([indexParams[, searchParams]])
indexParams : 인덱스 파라미터, 딕셔너리
algorithm : 알고리즘 선택 키
searchParams : 검색 파라미터, 딕셔너리 객체
checks = 32 : 검색할 후보 수
eps = 0 : 사용안함
sorted = Ture : 정렬해서 반환

```python
import cv2, numpy as np

img1 = cv2.imread('./img/yungyung.png')
img2 = cv2.imread('./img/3yung.png')
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# SIFT 생성
detector = cv2.xfeatures2d.SIFT_create()
# 키 포인트와 서술자 추출
kp1, desc1 = detector.detectAndCompute(gray1, None)
kp2, desc2 = detector.detectAndCompute(gray2, None)

# 인덱스 파라미터와 검색 파라미터 설정 ---①
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

# Flann 매처 생성 ---③
matcher = cv2.FlannBasedMatcher(index_params, search_params)
# 매칭 계산 ---④
matches = matcher.match(desc1, desc2)
# 매칭 그리기
res = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, \
                flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

plt.figure(figsize = (10,6))
imgs = {'Flann + SIFT':res}
for i, (k, v) in enumerate(imgs.items()):
    plt.subplot(1,1,i+1)
    plt.title(k)
    plt.imshow(v[:,:,(2,1,0)])
    plt.xticks([]),plt.yticks([])

plt.show()
```

```python
# ORB
import cv2, numpy as np

img1 = cv2.imread('./img/yungyung.png')
img2 = cv2.imread('./img/3yung.png')
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# ORB 추출기 생성
detector = cv2.ORB_create()
# 키 포인트와 서술자 추출
kp1, desc1 = detector.detectAndCompute(gray1, None)
kp2, desc2 = detector.detectAndCompute(gray2, None)

# 인덱스 파라미터 설정 ---①
FLANN_INDEX_LSH = 6
index_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6,
                   key_size = 12,
                   multi_probe_level = 1)
# 검색 파라미터 설정 ---②
search_params=dict(checks=32)
# Flann 매처 생성 ---③
matcher = cv2.FlannBasedMatcher(index_params, search_params)
# 매칭 계산 ---④
matches = matcher.match(desc1, desc2)
# 매칭 그리기
res = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, \
            flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
# 결과 출력            

plt.figure(figsize = (10,6))
imgs = {'Flann + ORB':res}
for i, (k, v) in enumerate(imgs.items()):
    plt.subplot(1,1,i+1)
    plt.title(k)
    plt.imshow(v[:,:,(2,1,0)])
    plt.xticks([]),plt.yticks([])

plt.show()
```

4.4 좋은 매칭점 찾기
앞서 살펴본 매칭 결과는 잘못된 정보를 너무 많이 포함하는 것을 알 수 있습니다. 그래서 매칭 결과에서 쓸모 없는 매칭점은 버리고 좋은 매칭점만을 골라내는 작업이 필요합니다.

```python
import cv2, numpy as np

img1 = cv2.imread('./img/yungyung.png')
img2 = cv2.imread('./img/3yung.png')
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# ORB로 서술자 추출 ---①
detector = cv2.ORB_create()
kp1, desc1 = detector.detectAndCompute(gray1, None)
kp2, desc2 = detector.detectAndCompute(gray2, None)
# BF-Hamming으로 매칭 ---②
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = matcher.match(desc1, desc2)

# 매칭 결과를 거리기준 오름차순으로 정렬 ---③
matches = sorted(matches, key=lambda x:x.distance)
# 최소 거리 값과 최대 거리 값 확보 ---④
min_dist, max_dist = matches[0].distance, matches[-1].distance
# 최소 거리의 15% 지점을 임계점으로 설정 ---⑤
ratio = 0.2
good_thresh = (max_dist - min_dist) * ratio + min_dist
# 임계점 보다 작은 매칭점만 좋은 매칭점으로 분류 ---⑥
good_matches = [m for m in matches if m.distance < good_thresh]
print('matches:%d/%d, min:%.2f, max:%.2f, thresh:%.2f' \
        %(len(good_matches),len(matches), min_dist, max_dist, good_thresh))
# 좋은 매칭점만 그리기 ---⑦
res = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, \
                flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
# 결과 출력

plt.figure(figsize = (10,6))
imgs = {'Good Match':res}
for i, (k, v) in enumerate(imgs.items()):
    plt.subplot(1,1,i+1)
    plt.title(k)
    plt.imshow(v[:,:,(2,1,0)])
    plt.xticks([]),plt.yticks([])

plt.show()
```

```python
# knnMatch 함수로 좋은 매칭점 찾기 ( k개의 최근접 이웃 매칭점을 더 가까운 순서대로 반환 )
import cv2, numpy as np

img1 = cv2.imread('./img/yungyung.png')
img2 = cv2.imread('./img/3yung.png')
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# ORB로 서술자 추출 ---①
detector = cv2.ORB_create()
kp1, desc1 = detector.detectAndCompute(gray1, None)
kp2, desc2 = detector.detectAndCompute(gray2, None)
# BF-Hamming 생성 ---②
matcher = cv2.BFMatcher(cv2.NORM_HAMMING2)
# knnMatch, k=2 ---③
matches = matcher.knnMatch(desc1, desc2, 2)

# 첫번재 이웃의 거리가 두 번째 이웃 거리의 75% 이내인 것만 추출---⑤
ratio = 0.75
good_matches = [first for first,second in matches \
                    if first.distance < second.distance * ratio]
print('matches:%d/%d' %(len(good_matches),len(matches)))

# 좋은 매칭만 그리기
res = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, \
                    flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
# 결과 출력                    

plt.figure(figsize = (10,6))
imgs = {'Matching':res}
for i, (k, v) in enumerate(imgs.items()):
    plt.subplot(1,1,i+1)
    plt.title(k)
    plt.imshow(v[:,:,(2,1,0)])
    plt.xticks([]),plt.yticks([])

plt.show()
```

총 368개의 매칭점 중에 9개의 좋은 매칭점을 찾은 것을 확인할 수 있습니다.

4.5 매칭 영역 원근 변환
좋은 매칭점으로만 구성된 매칭점 좌표들로 두 영상 간의 원근 변환행렬을 구하면 찾는 물체가 영상 어디에 있는지 표시할 수 있습니다. 이 과정에서 좋은 매칭점 중에 원근 변환행렬에 들어 맞지 않는 매칭점을 구분할 수 있어서 나쁜 매칭점을 또 한번 제거할 수 있습니다.

mtrx, mask = cv.findHomography(srcPoints, dstPoints[, method[, ransacReprojThreshold[, mask[, maxIters[, confidence]]]])
srcPoints : 원본 좌표 배열
dstPoints : 결과 좌표 배열
method = 0 : 근사 계산 알고리즘 선택 ( 0, cv2.RANSAC, cv2.LMEDS, cv2.RHO )
ransacReprojThreshold = 3 : 정상치 거리 임계 값
maxIters = 2000 : 근사 계산 반복 횟수
confidence : 신뢰도
mtrx : 결과 변환행렬
mask : 정상치 판별 결과
dst = cv.perspectiveTransform(src, m[, dst])
src : 입력 좌표 배열
m : 변환 행렬
dst : 출력 좌표 배열
cv2.findHomography()는 여러 개의 점으로 근사 계산한 원근 변환행렬을 반환합니다.
cv2.RANSAC(Random Sample Consensus) : 모든 입력점을 사용하지 않고 임의의 점들을 선정해서 만족도를 구하는 것을 반복해서 만족도가 가장 크게 선정된 점들만으로 근사 계산합니다.
cv2.LMEDS(Least Median of Squares) : 제곱의 최소 중간값을 사용합니다. 이 방법은 추가 파라미터를 요구하지 않아 사용하기 편리하지만, 정상치가 50% 이상 있는 경우에만 정상적으로 동작합니다.
cv2.RHD : 이상치가 많은 경우 더 빠른 방법입니다.

```python
import cv2, numpy as np

img1 = cv2.imread('./img/yungyung.png')
img2 = cv2.imread('./img/3yung.png')
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# ORB, BF-Hamming 로 knnMatch  ---①
detector = cv2.ORB_create()
kp1, desc1 = detector.detectAndCompute(gray1, None)
kp2, desc2 = detector.detectAndCompute(gray2, None)
matcher = cv2.BFMatcher(cv2.NORM_HAMMING2)
matches = matcher.knnMatch(desc1, desc2, 2)

# 이웃 거리의 75%로 좋은 매칭점 추출---②
ratio = 0.75
good_matches = [first for first,second in matches \
                    if first.distance < second.distance * ratio]
print('good matches:%d/%d' %(len(good_matches),len(matches)))

# 좋은 매칭점의 queryIdx로 원본 영상의 좌표 구하기 ---③
src_pts = np.float32([ kp1[m.queryIdx].pt for m in good_matches ])
# 좋은 매칭점의 trainIdx로 대상 영상의 좌표 구하기 ---④
dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good_matches ])
# 원근 변환 행렬 구하기 ---⑤
mtrx, mask = cv2.findHomography(src_pts, dst_pts)
# 원본 영상 크기로 변환 영역 좌표 생성 ---⑥
h,w, = img1.shape[:2]
pts = np.float32([ [[0,0]],[[0,h-1]],[[w-1,h-1]],[[w-1,0]] ])
# 원본 영상 좌표를 원근 변환  ---⑦
dst = cv2.perspectiveTransform(pts,mtrx)
# 변환 좌표 영역을 대상 영상에 그리기 ---⑧
img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

# 좋은 매칭 그려서 출력 ---⑨
res = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, \
                    flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

plt.figure(figsize = (10,6))
imgs = {'Matching Homography':res}
for i, (k, v) in enumerate(imgs.items()):
    plt.subplot(1,1,i+1)
    plt.title(k)
    plt.imshow(v[:,:,(2,1,0)])
    plt.xticks([]),plt.yticks([])

plt.show()
```

good_matches에 있는 매칭점과 같은 자리에 있는 키 포인트 객체에서 각 영상의 매칭점 좌표를 구합니다.

```python
# 매칭기를 통해 얻은 모든 매칭점을 RANSAC 원근 변환 근사 계산으로 잘못된 매칭을 가려냅니다.
import cv2, numpy as np
import matplotlib.pyplot as plt

img1 = cv2.imread('./img/yungyung.png')
img2 = cv2.imread('./img/3yung.png')
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# ORB, BF-Hamming 로 knnMatch  ---①
detector = cv2.ORB_create()
kp1, desc1 = detector.detectAndCompute(gray1, None)
kp2, desc2 = detector.detectAndCompute(gray2, None)
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = matcher.match(desc1, desc2)

# 매칭 결과를 거리기준 오름차순으로 정렬 ---③
matches = sorted(matches, key=lambda x:x.distance)
# 모든 매칭점 그리기 ---④
res1 = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, \
                    flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

# 매칭점으로 원근 변환 및 영역 표시 ---⑤
src_pts = np.float32([ kp1[m.queryIdx].pt for m in matches ])
dst_pts = np.float32([ kp2[m.trainIdx].pt for m in matches ])
# RANSAC으로 변환 행렬 근사 계산 ---⑥
mtrx, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
h,w = img1.shape[:2]
pts = np.float32([ [[0,0]],[[0,h-1]],[[w-1,h-1]],[[w-1,0]] ])
dst = cv2.perspectiveTransform(pts,mtrx)
img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

# 정상치 매칭만 그리기 ---⑦
matchesMask = mask.ravel().tolist()
res2 = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, \
                    matchesMask = matchesMask,
                    flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
# 모든 매칭점과 정상치 비율 ---⑧
accuracy=float(mask.sum()) / mask.size
print("accuracy: %d/%d(%.2f%%)"% (mask.sum(), mask.size, accuracy))

# 결과 출력                    

plt.figure(figsize = (8,8))
imgs = {'Matching-All':res1, 'Matching-Inlier ':res2}
for i, (k, v) in enumerate(imgs.items()):
    plt.subplot(2,1,i+1)
    plt.title(k)
    plt.imshow(v[:,:,(2,1,0)])
    plt.xticks([]),plt.yticks([])

plt.show()
```

89개의 전체 매칭점 중에 근사 계산에서 정상치로 판단한 매칭점은 8개입니다. 정상치 매칭점 개수 자체만으로도 그 수가 많을수록 원본 영상과의 정확도가 높다고 볼 수 있고 전체 매칭점 수와의 비율이 높으면 더 확실하다고 볼 수 있습니다.


