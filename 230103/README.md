# 230103~
json 코드 확인: https://jsoneditoronline.org/#left=local.mikuzi

data: \\192.168.0.205\data1\Xray

opencv: https://076923.github.io/posts/Python-opencv-1/

## 목차
image 폴더는 원본 데이터
하나의 이미지는 low, high로 나눠져 crop 폴더에 저장되고, 해당 객체가 맞춰 저장

1. crop 데이터 low, high 세트로 width, hight가 똑같이 늘리기 -> 한 세트 당 3번
2. 해당 데이터가 변경되었을 때 crop_data.json도 변경

### crop_data.json 구조
#### 1. image: crop 데이터 정보
1. id: 1부터 계속 늘어남
2. dataset_id: 계속 1
3. path: 파일 위치 정보
4. file_name: 파일 이름
5. width, height(width, height는 low, high 세트는 같다)

#### 2. categories
1. id: id
2. name: name

#### 3. annotations
1. id: image 아이디
2. image_id: image 아이디
3. category_id: categories의 정보
4. segmentation: 해당 사진 데이터 점들 x,y 정보가 차례로 저장
5. area: w*h
6. bbox: x, y, w, h


-> 주의할 점 이미지 16bit로 불러들이고 저장해야함

# 230104 ~ 230105
230103에서
1. bbox 추가: cv2.rectangle 사용해서
2. segmentation 투명하게: https://hazel-developer.tistory.com/78


주어진 file이랑 json images의 path 순서가 다름..
일단 이미지 파일 순서대로 low, high 세트이기 때문에 이미지 파일의 이름을 기준으로 데이터를 늘려야함.
그리고 json에서 해당 이미지 파일 이름을 기준으로 path 정보를 가져와서 사용해야함.

# 230106
+)
1. xray_officeutilityknife_a_1: 1부터 시작함(o)
2. xray_scissors_1: 숫자가 중구난방
3. xray_laserpointer_f_1: 30, 31이 없음(o)
위에 맞게 코드를 또 수정해야함..