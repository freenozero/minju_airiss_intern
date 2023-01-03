# 230103~
json 코드 확인: https://jsoneditoronline.org/#left=local.mikuzi
data: \\192.168.0.205\data1\Xray

## 목차
image는 원본 데이터
하나의 이미지는 low, high로 나눠져 crop 파일에 저장되고, 해당 객체가 잘려 저장

### -> crop 데이터 증가 시키기
1. crop 데이터 low, high 세트로 width, hight가 똑같이 늘리기 -> 한 세트 당 3번
2. 해당 데이터가 변경되었을 때 crop_data.json도 변경

* json 파일 변경 시에 주의 할 점 *
