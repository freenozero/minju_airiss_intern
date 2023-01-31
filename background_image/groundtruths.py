import json
import os
import cv2
import numpy as np
import random
import natsort

def main():
    groundTruths()

# json 불러오기
def jsonLoad(json_path):
    with open(json_path) as json_file:
        json_data = json.load(json_file)

    return json_data

# png 파일만 불러오기
def pngLoad(file_path):
    file = [f for f in os.listdir(file_path) if f.endswith('.png')]
    file = natsort.natsorted(file)  # 정렬
    return file

def groundTruths():
    origin_img_path = "D:/wp/data/background_manipulation/manipulation/manipulation_image/image"
    json_path = "D:/wp/data/background_manipulation/manipulation/manipulation_image/json/data.json"
    ground_truths_path = "D:/wp/data/background_manipulation/manipulation/manipulation_image/ground_truths"

    origin_files = pngLoad(origin_img_path)

    json_data = jsonLoad(json_path)
    
    annotations = json_data['annotations']
    
    # 이전 이미지 이름 저장
    before_image_name = 0
    category_color = [(255, 0, 0), (0, 0, 255), (255, 255, 0), (0, 255, 0)]

    #annotaion 불러오기
    for annotation in annotations:
        image_id = annotation["image_id"]
        category_id = (annotation["category_id"])
        image_name = origin_files[image_id]

        if(before_image_name != image_name):
            original_img = cv2.imread(f"{origin_img_path}/{image_name}", 1)
            ground_truths_img = original_img.copy()
            # print(ground_truths_img.shape)
            before_image_name = image_name

        seg = []
        for index, x in enumerate(annotation['segmentation'][0][0]):
            if((index % 2) == 0):
                seg.append([x, annotation['segmentation'][0][0][index+1]])
        seg = np.array(seg, np.int32)

        # bbox 불러오기
        bbox = annotation['bbox']

        # seg 칠하기
        cv2.fillPoly(ground_truths_img, [seg], category_color[category_id])

        # bbox 그리기
        cv2.rectangle(ground_truths_img, (bbox[0], bbox[1]),
                      (bbox[2], bbox[3]), category_color[category_id], 3)

        # original_img랑 filter_img 합성하기
        add_img = cv2.addWeighted(original_img, 0.7, ground_truths_img, 0.3, 3)

        # add_image 폴더 없을 시 생성
        if not os.path.exists(ground_truths_path):
            os.makedirs(ground_truths_path)
        

        # 합성한 이미지 저장
        if(before_image_name == image_name):
            # cv2.imshow("add_img",  add_img)
            # cv2.waitKey(0)
            cv2.imwrite(f"{ground_truths_path}/{image_name}", add_img)

if __name__ == "__main__":
    main()