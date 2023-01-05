import json
import os
import cv2
import numpy as np
import random
import natsort
import os


def augmentation(file_path, json_path):
    file = os.listdir(file_path)
    file = natsort.natsorted(file)  # 정렬 (파일 불러오면 1, 10, 11 ... list 저장 해결법)
    data_max = len(file)-1  # -1은 Thumbs.db 제외

    # json 파일 불러오기
    with open(json_path) as json_file:
        json_data = json.load(json_file)

    json_file_dic = {}
    for i in range(len(json_data['images'])):
        file_name = json_data['images'][i]['file_name']
        if (file_name != 'Thumbs.db'):
            json_file_dic[file_name] = i

    for _ in range(0, 1):
        for f in file:
            if (f != 'Thumbs.db'):
                # json 파일 불러오기
                with open(json_path) as json_file:
                    json_data = json.load(json_file)

                file_name = int(f.rstrip('.png'))
                # -1은 16 bit 이미지 불러오기 위함
                img = cv2.imread(file_path + '/' + f, -1)
                # file 이름이 짝수일때마다 random 생성
                # update_fx와 값을 다르게 주기 위해 if문 두개
                if (file_name % 2 == 0):
                    # x update: img.shape[0] / update_img.shape[0]
                    update_cols = round(random.uniform(0.5, 1.5), 1)
                if (file_name % 2 == 0):
                    update_rows = round(random.uniform(0.5, 1.5), 1)  # y update
                # 상대적인 비율로 축소, 확대
                update_img = cv2.resize(img, dsize=(
                    0, 0), fx=update_cols, fy=update_rows, interpolation=cv2.INTER_LINEAR)
                # print('원본(height, width):', img.shape)
                # print('업데이트(height, width):', update_img.shape)
                # print('fx, fy:', update_cols, update_rows)

                # segmentation update: seg(x) * update_rows, seg(y) * update_rows
                new_seg = list()
                num = 0

                for i in json_data['annotations'][json_file_dic[f]]['segmentation'][0][0]:
                    if ((num % 2) == 0):
                        new_seg.append(round(i * update_cols, 1))
                    else:
                        new_seg.append(round(i * update_rows, 1))
                    # print(new_seg)
                    num += 1

                # data_max 값에 추가
                new_annotations = {'id': data_max+2,
                                'image_id': data_max+2,
                                'category_id': json_data['annotations'][json_file_dic[f]]['category_id'],
                                # x, y, width, height
                                'bbox': [0, 0, update_img.shape[1], update_img.shape[0]],
                                'segmentation': [[new_seg]],
                                # height * width
                                'area': update_img.shape[0]*update_img.shape[1],
                                'iscrowd': False,
                                'color': 'Unknown',
                                'unitID': 1,
                                'registNum': 1,
                                'number1': 4,
                                'number2': 4,
                                'weight': None}
                json_data['annotations'].append(new_annotations)

                new_images = {'id': data_max+2,
                            'dataset_id': 1,
                            'path': file_path + '/' + str(data_max) + '.png',
                            'file_name': str(data_max) + '.png',
                            'width': update_img.shape[1],
                            'height': update_img.shape[0]}
                json_data['images'].append(new_images)

                # 이미지 보기
                # cv2.imshow('img', img)  # 이전
                # cv2.imshow('update_img', update_img)
                # cv2.waitKey(0)

                # print(new_annotations)
                # print(new_images)

                # seg 겹쳐서 이미지 보기
                # seg = []
                # for j in range(0, len(new_seg)):
                #     if j % 2 == 0:
                #         seg.append((new_seg[j], new_seg[j+1]))
                # print(seg)
                # seg = np.array(seg, np.int32)
                # cv2.fillConvexPoly(update_img, seg, color=(255,0,0))
                # cv2.imshow("img", update_img)
                # cv2.waitKey(0)

                # json 파일 저장하기
                with open(json_path, 'w') as json_file:
                    json_data = json.dump(json_data, json_file)

                # 이미지 파일 저장하기
                cv2.imwrite(file_path + '/' + str(data_max+1) + '.png', update_img)
                data_max += 1


def filter_image(json_path, origin_img_path, file_path, add_img_path):
    origin_file = os.listdir(origin_img_path)
    origin_file = natsort.natsorted(origin_file)  # 정렬 (파일 불러오면 1, 10, 11 ... list 저장 해결법)
    len_origin_file = len(origin_file)-1  # -1은 Thumbs.db 제외

    file = os.listdir(file_path)
    file = natsort.natsorted(file)  # 정렬 (파일 불러오면 1, 10, 11 ... list 저장 해결법)
    len_file = len(file)-1  # -1은 Thumbs.db 제외

   # json 파일 불러오기
    with open(json_path) as json_file:
        json_data = json.load(json_file)

    json_file_dic = {}
    for i in range(len(json_data['images'])):
        file_name = json_data['images'][i]['file_name']
        if (file_name != 'Thumbs.db'):
            json_file_dic[file_name] = i

    for i in range(len_origin_file+1, len_file+1):
        img = cv2.imread(file_path + '/' + file[i], 1)

        # segmentation 불러오기
        seg = []
        for j in range(0, len(json_data['annotations'][i]['segmentation'][0][0])):
            if j % 2 == 0:
                seg.append(([json_data['annotations'][i]['segmentation'][0][0]
                           [j], json_data['annotations'][i]['segmentation'][0][0][j+1]]))
        seg = np.array(seg, np.int32)
        
        # bbox 불러오기
        bbox = json_data['annotations'][i]['bbox']

        # seg 칠하기
        filter_img = np.full(
            (img.shape[0], img.shape[1], 3), (0, 0, 0), dtype=np.uint8)
        cv2.fillConvexPoly(filter_img, seg, color=(0, 0, 255, 1))

        # bbox 그리기
        cv2.rectangle(filter_img, (bbox[0], bbox[1]),
                      (bbox[2], bbox[3]), (255, 0, 0, 1), 3)


        # img랑 seg_img 합성하기
        add_img = cv2.addWeighted(img, 0.7, filter_img, 0.3, 0)

        # 합성한 이미지 보기
        # cv2.imshow("add_img", add_img)
        # cv2.waitKey(0)

        # 합성한 이미지 저장
        if not os.path.exists(add_img_path):
            os.makedirs(add_img_path)
        cv2.imwrite(add_img_path + '/' + str(i) + '.png', add_img)

# img_name만 변경하면 됨.
img_name = "xray_jackknife_a_2"

file_path = 'D:/wp/data/' + img_name + '/crop'
json_path = 'D:/wp/data/' + img_name + '/json/crop_data.json'
origin_img_path = 'D:/wp/data/원본/' + img_name + '/crop'
add_img_path = 'D:/wp/data/' + img_name + '/add_image'

augmentation(file_path, json_path)
filter_image(json_path, origin_img_path, file_path, add_img_path)
