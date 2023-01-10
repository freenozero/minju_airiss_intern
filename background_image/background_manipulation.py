import json
import os
import cv2
import numpy as np
import random
import natsort

def main():
    manipulation()

# json 불러오기
def json_load(json_path):
    with open(json_path) as json_file:
        json_data = json.load(json_file)

    json_file_dic = {}
    # file_name = i 형식
    for i in range(len(json_data['images'])):
        file_name = json_data['images'][i]['file_name']
        json_file_dic[file_name] = i

    return json_data, json_file_dic

# json 저장
def json_dump(json_path, json_data):
    with open(json_path, 'w') as json_file:
        json_data = json.dump(json_data, json_file)

# png 파일만 불러오기
def png_load(file_path):
    file = [f for f in os.listdir(file_path) if f.endswith('.png')]
    file = natsort.natsorted(file)  # 정렬
    return file

# 이미지 합성하기
def manipulation():
    categorical = ["knife", "gun", "bettery", "laserpointer"]
    
    background_path = "D:/wp/data/background_manipulation/manipulation/background_image"
    background_file = png_load(background_path)

    for i, category in enumerate(categorical):
        image_path = f"D:/wp/data/background_manipulation/manipulation/categorical_image/{category}/crop"
        image_file = png_load(image_path)
        
        json_path = f"D:/wp/data/background_manipulation/manipulation/categorical_image/{category}/json/crop_data.json"
        json_data, json_dict = json_load(json_path)

        print(json_data)

if __name__ == "__main__":
    main()