import json
import os
import cv2
import numpy as np
import random
import natsort
import time
import math

# json 불러오기
def jsonLoad(json_path):
    with open(json_path) as json_file:
        json_data = json.load(json_file)

    json_file_dic = {}
    # file_name = i 형식
    for i in range(len(json_data['images'])):
        file_name = json_data['images'][i]['file_name']
        json_file_dic[file_name] = i
    
    json_file = []
    json_file.append(json_data)
    json_file.append(json_file_dic)

    return json_file

# json 저장
def jsonDump(json_path, json_data):
    with open(json_path, 'w') as json_file:
        json_data = json.dump(json_data, json_file)

# png 파일만 불러오기
def pngLoad(file_path):
    file = [f for f in os.listdir(file_path) if f.endswith('.png')]
    file = natsort.natsorted(file)
    return file

# 모든 필요한 이미지 불러오기
def allFileLoad():
    categorical = ["knife", "gun", "bettery", "laserpointer"]
    image_file, json_file, background_file = [], [], []
    background_path = "D:/wp/data/background_manipulation/manipulation/background_image"

    # 모든 이미지 불러오기
    for category in categorical:
        image_path = f"D:/wp/data/background_manipulation/manipulation/categorical_image/{category}/crop"
        image_file.append(pngLoad(image_path))

        json_path = f"D:/wp/data/background_manipulation/manipulation/categorical_image/{category}/json/crop_data.json"
        json_file.append(jsonLoad(json_path))
    background_file = pngLoad(background_path)

    return image_file, json_file, background_file

# 이미지 index 랜덤 추출
def itemIndexRandom(item_index, item_used):
    # 모든 이미지가 0개가 아닐 때까지 추출
    while (listLen(item_index) == 0):
        # 카테고리 for문
        for i in range (0, len(item_index)):
            # use image cnt
            item_use_cnt = random.randint(0, 2)
            
            # 카테고리당 item_use_cnt만큼 뽑기
            for _ in range(0, item_use_cnt):
                index = random.randrange(0, 9999, 2)
                # 사용하지 않은 a가 나올 때 까지
                while (index in item_used[i]):
                    # 10,000장을 다 사용시
                    if (len(item_used[i]) >= 10000):
                        item_used[i] = []
                        
                    index = random.randrange(0, 9999, 2)
                item_index[i].append(index)

    # used image save
    item_used = itemUsedSave(item_index, item_used)
    return item_index, item_used

# 사용한 이미지 저장
def itemUsedSave(item_index, item_used):
    for i in range(4):
        # 이미지가 []이지 않을 때 저장
        if item_index[i] != []:
            item_used[i] += item_index[i]
    return item_used

# list length
def listLen(l):
    return len(l[0])+len(l[1])+len(l[2])+len(l[3])

def main():
    image_file, json_file, background_file = allFileLoad()

    # 사용한 이미지 저장 리스트
    item_used = [[],[],[],[]]

    for file_name in range(0, 19999):
        # print(file_name)
        # print(listLen(item_used))
        # print(len(item_used[0]), len(item_used[1]),len(item_used[2]),len(item_used[3]))
        
        # 사용할 이미지 index 저장 리스트
        item_index = [[],[],[],[]]

        # background index
        bk_random_index = random.randrange(0, len(background_file), 2)

        # high
        # background
        bk_image = background_file[bk_random_index]
        # itemImage
        item_index, item_used = itemIndexRandom(item_index, item_used)
        # item bk image 불러오기

        # low  
        # background
        bk_image = background_file[bk_random_index+1]
        # itemImage
        item_index, item_used = itemIndexRandom(item_index, item_used)
        

if __name__ == "__main__":
    main()