import json
import os
import cv2
import numpy as np
import random
import natsort 

def augmentation(file_path, json_path):
    file = os.listdir(file_path)
    file = natsort.natsorted(file) # 정렬 (파일 불러오면 1, 10, 11 ... list 저장 해결법)
    data_max = len(file) - 1  # -1은 Thumbs.db 제외
    print(data_max)

    # json 파일 불러오기
    with open(json_path) as f:
        json_data = json.load(f)

    # 한 세트(row, high당 3번 업데이트)
    for _ in range(0, 3):
        # 파일 하나씩 불러오기
        for f in file:
            file_name = int(f.rstrip('.png'))
            # Thumbs.db는 제외
            if (f != 'Thumbs.db'):
                # file 이름이 짝수일때마다 random 생성
                
                #-1은 16 bit 이미지 불러오기 위함
                img = cv2.imread(file_path + '/' + f, -1)

                #update_fx와 값을 다르게 주기 위해 if문 두개
                if (file_name % 2 == 0):
                    update_cols = round(random.uniform(0.5, 1.5), 1) #x update: img.shape[0] / update_img.shape[0]
                if (file_name % 2 == 0):
                    update_rows = round(random.uniform(0.5, 1.5), 1) #y update

                # 상대적인 비율로 축소, 확대
                update_img = cv2.resize(img, dsize=(0, 0), fx=update_cols, fy=update_rows, interpolation=cv2.INTER_LINEAR)
                print('원본(height, width):', img.shape)
                print('업데이트(height, width):', update_img.shape)
                print('fx, fy:', update_cols, update_rows)
                
                cv2.imshow('img', img)
                cv2.imshow('update_img', update_img)
                cv2.waitKey(0)

                #segmentation update: seg(x) * update_rows, seg(y) * update_rows
                new_seg = list()
                for i in json_data['annotations'][file_name]['segmentation'][0][0]:
                    num = 0
                    if((num%2) == 0):
                        new_seg.append(round(i * update_rows,1))
                    else:
                        new_seg.append(round(i * update_cols, 1))
                    num += 1
                
                print(json_data['annotations'][file_name]['segmentation'])
                print(new_seg)
                
               # json_data['annotations'][file_name]['segmentation'][0][0] = new_seg

                #data_max 값에 추가
                new_annotations = {'id': data_max,
                                      'image_id': data_max,
                                      'category_id': json_data['annotations'][file_name]['category_id'],
                                      'bbox': [0, 0, update_img.shape[1], update_img.shape[0]], #x, y, width, height
                                      'segmentation': [[[new_seg]]],
                                      'area': update_img.shape[0]*update_img.shape[1], #height * width
                                      'iscrowd': False,
                                      'color': 'Unknown',
                                      'unitID': 1,
                                      'registNum': 1,
                                      'number1': 4,
                                      'number2': 4,
                                      'weight': None}
                json_data['annotations'].append(new_annotations)

                new_images = {'id': data_max,
                                  'dataset_id': data_max,
                                  'path': file_path + '/' + str(data_max) + '.png',
                                  'file_name': str(data_max) + '.png',
                                  'width': update_img.shape[1],
                                  'height': update_img.shape[0]}
                json_data['images'].append(new_images)

                print(new_annotations)
                print(new_images)

                #json 파일 저장하기 
                print(json_path)
                with open(json_path, 'w') as f:
                    json_data = json.dump(json_data, f)

                #이미지 파일 저장하기
                cv2.imwrite(file_path + '/' + str(data_max) + '.png', update_img)
                data_max += 1



file_path = 'D:/wp/data/xray_artknife_a_1/crop'
json_path = 'D:/wp/data/xray_artknife_a_1/json/crop_data.json'
augmentation(file_path, json_path)
