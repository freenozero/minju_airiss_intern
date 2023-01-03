import json
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

file_path = 'D:/wp/data/xray_artknife_a_1/crop'

file = os.listdir(file_path)

#random
random_width, random_height = 0, 0

#한 세트(row, high당 3번 업데이트)
for i in range(0, 3):
    #파일 하나씩 불러오기
    for f in file:
        #Thumbs.db는 제외
        if(f != 'Thumbs.db'):
            #file 이름이 짝수일때마다 random 생성
            if(int(f.rstrip('.png')) % 2 == 0):
                update_width = random.uniform(0.5, 5)
                update_height = random.uniform(0.5, 5)
                #print(update_width, update_height)
    
            img = cv2.imread(file_path + '/' + f, -1)
            
            #상대적인 비율로 축소, 확대
            update_img = cv2.resize(img, dsize=(0, 0), fx=update_width, fy=update_height, interpolation=cv2.INTER_LINEAR)
            #print(update_img.shape)

            cv2.imshow('img', img)
            cv2.imshow('update_img', update_img)
            cv2.waitKey(0)

            #json 파일 불러오기