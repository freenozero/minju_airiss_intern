import json
import os
import cv2
import numpy as np
import random
import natsort


def augmentation(file_path, json_path, img_name):
    file = os.listdir(file_path)
    if "Thumbs.db" in file:
        file.remove("Thumbs.db")
    file = natsort.natsorted(file)  # 정렬 (파일 불러오면 1, 10, 11 ... list 저장 해결법)

    # json 파일 불러오기
    with open(json_path) as json_file:
        json_data = json.load(json_file)

    json_file_dic = {}
    for i in range(len(json_data['images'])):
        file_name = json_data['images'][i]['file_name']
        json_file_dic[file_name] = i

    save_file_name = file[len(file)-1].rstrip('.png')
    
    save_file_name = str(int(save_file_name)+ 1) 
    for _ in range(0, 1):
        for f in file:
            # json 파일 불러오기
            with open(json_path) as json_file:
                json_data = json.load(json_file)
            
            file_name = int(f.rstrip('.png'))
            # print(len(json_data['images']) + diff, file_name)

            # -1은 16 bit 이미지 불러오기 위함
            img = cv2.imread(file_path + '/' + f, -1)
            # file 이름이 짝수일때마다 random 생성
            # update_fx와 값을 다르게 주기 위해 if문 두개

            # 이미지 시작 파일이 짝수인지 홀수인지 구별 후 random 
            if ((int(file[0].rstrip('.png'))%2) == 0): 
                if ((file_name) % 2 == 0):
                    # x update: img.shape[0] / update_img.shape[0]
                    update_cols = round(random.uniform(0.5, 1.5), 1)
                if ((file_name) % 2 == 0):
                    update_rows = round(random.uniform(0.5, 1.5), 1)  # y update
            else:
                if ((file_name) % 2 != 0):
                    update_cols = round(random.uniform(0.5, 1.5), 1)
                if ((file_name) % 2 != 0):
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
            
            #seg
            for i in json_data['annotations'][json_file_dic[f]]['segmentation'][0][0]:
                if ((num % 2) == 0):
                    new_seg.append(round(i * update_cols, 1))
                else:
                    new_seg.append(round(i * update_rows, 1))
                # print(new_seg)
                num += 1

            if "xray_scissors" in img_name:
                save_file_name.zfill(5)

            # json 파일 추가
            new_images = {'id': len(json_data['images']) +1,
                            'dataset_id': json_data['images'][json_file_dic[f]]['dataset_id'],
                            'path': file_path + '/' + save_file_name + '.png',
                            'file_name': save_file_name + '.png',
                            'width': update_img.shape[1],
                            'height': update_img.shape[0]}
            json_data['images'].append(new_images)

            new_annotations = {'id': len(json_data['images']),
                                'image_id': len(json_data['images']),
                                'category_id': json_data['annotations'][json_file_dic[f]]['category_id'],
                                # x, y, width, height
                                'bbox': [0, 0, update_img.shape[1], update_img.shape[0]],
                                'segmentation': [[new_seg]],
                                # height * width
                                'area': update_img.shape[0]*update_img.shape[1],
                                'iscrowd': json_data['annotations'][json_file_dic[f]]['iscrowd'],
                                'color': json_data['annotations'][json_file_dic[f]]['color'],
                                'unitID': json_data['annotations'][json_file_dic[f]]['unitID'],
                                'registNum': json_data['annotations'][json_file_dic[f]]['registNum'],
                                'number1': json_data['annotations'][json_file_dic[f]]['number1'],
                                'number2': json_data['annotations'][json_file_dic[f]]['number2'],
                                'weight': json_data['annotations'][json_file_dic[f]]['weight']}
            json_data['annotations'].append(new_annotations)

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
            # print(new_seg)
            # seg = np.array(seg, np.int32)
            # cv2.fillConvexPoly(update_img, seg, color=(255,0,0))
            # cv2.imshow("img", update_img)
            # cv2.waitKey(0)

            # 이미지 파일 저장
            cv2.imwrite(file_path + '/' + save_file_name + '.png', update_img)

            # json 파일 저장
            with open(json_path, 'w') as json_file:
                json_data = json.dump(json_data, json_file)
            
            file_name += 1
            save_file_name = str(int(save_file_name)+ 1) 

            if "xray_scissors" in img_name:
                save_file_name = save_file_name.zfill(5)
            


def filter_image(json_path, origin_img_path, file_path, add_img_path):
    origin_file = os.listdir(origin_img_path)
    # 정렬 (파일 불러오면 1, 10, 11 ... list 저장 해결법)
    origin_file = natsort.natsorted(origin_file)

    file = os.listdir(file_path)
    file = natsort.natsorted(file)  # 정렬 (파일 불러오면 1, 10, 11 ... list 저장 해결법)

   # json 파일 불러오기
    with open(json_path) as json_file:
        json_data = json.load(json_file)

    json_file_dic = {}
    for i in range(0, len(json_data['images'])):
        file_name = json_data['images'][i]['file_name']
        json_file_dic[file_name] = i

    # origin_file에 없는 file 값들 저장
    s = set(origin_file)
    file = [x for x in file if x not in s]
    
    for i in file:
        img = cv2.imread(file_path + '/' + i, 1)

        # segmentation 불러오기
        seg = []
        file_name = str(i.rstrip('.png'))
        for j in range(0, len(json_data['annotations'][json_file_dic[i]]['segmentation'][0][0])):
            if j % 2 == 0:
                seg.append(([json_data['annotations'][json_file_dic[i]]['segmentation'][0][0]
                           [j], json_data['annotations'][json_file_dic[i]]['segmentation'][0][0][j+1]]))
        seg = np.array(seg, np.int32)

        # bbox 불러오기
        bbox = json_data['annotations'][json_file_dic[i]]['bbox']

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
        cv2.imwrite(add_img_path + '/' + i, add_img)

# img_name만 변경하면 됨.
img_name = "xray_scissors_1"

file_path = 'D:/wp/data/' + img_name + '/crop'
json_path = 'D:/wp/data/' + img_name + '/json/crop_data.json'
origin_img_path = 'D:/wp/data/원본/' + img_name + '/crop'
add_img_path = 'D:/wp/data/' + img_name + '/add_image'

augmentation(file_path, json_path, img_name)
filter_image(json_path, origin_img_path, file_path, add_img_path)
