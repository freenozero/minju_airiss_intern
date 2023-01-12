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

# 폴더에 모든 이미지 이름 불러오기
def FolderImgNameLoad(file_path):
    file = [f for f in os.listdir(file_path) if f.endswith('.png')]
    file = natsort.natsorted(file)  # 정렬
    return file

# 이미지 load
def pngLoad(file_path):
    file = cv2.imread(file_path, -1)
    return file

# item 모든 이미지 불러오기
def allItemLoad(item_index):
    #categorical
    categorical = ["knife", "gun", "bettery", "laserpointer"]
    item_images = [[],[],[],[]]
    # 모든 이미지 불러오기
    for i, category in enumerate(categorical):
        for file_name in item_index[i]:
            image_path = f"D:/wp/data/background_manipulation/manipulation/categorical_image/{category}/crop/{file_name}.png"
            item_images[i].append(pngLoad(image_path))
            
    return item_images

# 이미지 index 랜덤 추출
def itemIndexRandom(item_used, all_item_used):
    item_index = [[],[],[],[]]

    # 모든 이미지가 0개가 아닐 때까지 추출
    while (listLen(item_index) == 0):
        # 카테고리 for문
        for i in range (0, len(item_index)):
            # use image cnt
            item_use_cnt = random.randint(0, 2)
            # (확인용) 이미지 카테고리당 사용한 횟수 계산
            all_item_used[i] += item_use_cnt

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

def manipulation(bk_images, item_index):

    max_pixel = 65535 #16비트 max_pixel
    random_max = [[],[],[],[]]
    for i, bk_image in enumerate(bk_images):
        bk_image = (bk_image/max_pixel)
        
        # item all image 불러오기 [[],[],[],[]]
        # high
        if(i % 2 == 0):
            item_images = allItemLoad(item_index)
        # low
        else:
            matrix = [[1 for col in range(len(item_index[row]))] for row in range(len(item_index))]
            item_index = [[c + d for c, d in zip(a,b)] for a, b in zip(item_index, matrix)]
            item_images = allItemLoad(item_index)

        # 카테고리별로
        for j, categorical_images in enumerate(item_images):
            # 카테고리 안에서 하나씩
            for z, item_image in enumerate(categorical_images):

                item_image = (item_image/max_pixel)

                item_height, item_width = item_image.shape
                bk_height, bk_width = bk_image.shape

                # high
                if(i % 2 == 0):
                    random_max_height = random.randrange(item_height, bk_height, item_height)
                    random_max_width = random.randrange(item_width, bk_width, item_width)
                    random_max[j].append([random_max_height, random_max_width])
                    #오른쪽 하단: img1[height1-min_height: height1, width1-min_width:width1] = img_sub
                    #오른쪽 상단: img1[0:min_height, width1-min_width:width1] = img_sub
                    #왼쪽 하단: img1[height1-min_height: height1, 0:min_width] = img_sub
                    #왼쪽 상단: img1[0:min_height,0:min_width] = img_sub
                    bk_image[random_max_height-item_height:random_max_height, random_max_width-item_width:random_max_width] *= item_image
                    # print(random_max)
                # low
                else:
                    bk_image[random_max[j][z][0]-item_height:random_max[j][z][0], random_max[j][z][1]-item_width:random_max[j][z][1]] *= item_image
     
        bk_images[i] = bk_image
    return bk_images

def main():
    # path
    background_path =  "D:/wp/data/background_manipulation/manipulation/background_image"

    # image file name load
    background_file = FolderImgNameLoad(background_path)
        
    # 사용한 이미지 저장 리스트
    item_used = [[],[],[],[]]

    # (확인용) 이미지 카테고리당 사용한 횟수 계산
    all_item_used = [0 for i in range(4)]

    for file_name in range(0, 19999):
        # print(file_name)
        # print(listLen(item_used))
        # print(len(item_used[0]), len(item_used[1]),len(item_used[2]),len(item_used[3]))
        # print(all_item_used)


        # background index
        bk_random_index = random.randrange(0, 8, 2)
        # background image list
        bk_images = []
        # background
        bk_images.append(pngLoad(f"{background_path}/{background_file[bk_random_index]}"))
        bk_images.append(pngLoad(f"{background_path}/{background_file[bk_random_index+1]}"))

        # itemImage
        item_index, item_used = itemIndexRandom(item_used, all_item_used)
        # item 이미지 확인
        # for i in range(len(item_image)):
        #     for j in range(len(item_image[i])):
        #         print(item_image[i][j])
        #         cv2.imshow("image", item_image[i][j])
        #         cv2.waitKey(0)

        # image manipulation
        bk_images = manipulation(bk_images, item_index)
        cv2.imshow("manipulation_image_high", bk_images[0])
        cv2.imshow("manipulation_image_low", bk_images[1])
        cv2.waitKey(0)



        

if __name__ == "__main__":
    main()