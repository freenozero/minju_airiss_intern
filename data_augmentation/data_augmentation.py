import json
import os
import cv2
import numpy as np
import random
import natsort

def main():
    img_name = "xray_officeutilityknife_a_1"
    augmentation(img_name)
    filter_image(img_name)

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

def augmentation(img_name):
    file_path = f"D:/wp/data/{img_name}/crop"
    json_path = f"D:/wp/data/{img_name}/json/crop_data.json"

    file = png_load(file_path)
    json_data, json_file_dic = json_load(json_path)
    
    # 저장할 파일 이름을 crop 폴더 max에서 +1
    save_file_name = file[len(file)-1].rstrip('.png')
    #시작 파일 짝수 홀수
    first_file_name = int(file[0].rstrip('.png')) % 2

    # 한 장에 몇 번 저장할지
    for _ in range(0, 1):
        # 파일 하나씩 불러오기
        for f in file:
            file_name = int(f.rstrip('.png'))

            # -1은 16 bit 이미지 불러오기 위함
            img = cv2.imread(file_path + '/' + f, -1)

            if (first_file_name == 0 and file_name % 2 == 0): #시작 파일이 짝수고, 현재 증강할 파일이 짝수면 random
                    update_cols = round(random.uniform(0.5, 1.5), 1)
                    update_rows = round(random.uniform(0.5, 1.5), 1)  # y update
            elif (first_file_name != 0 and file_name % 2 != 0): #시작 파일이 홀수고, 현재 증강할 파일이 홀수면 random
                    update_cols = round(random.uniform(0.5, 1.5), 1)
                    update_rows = round(random.uniform(0.5, 1.5), 1)  # y update

            # 상대적인 비율로 축소, 확대
            update_img = cv2.resize(img, dsize=(
                0, 0), fx=update_cols, fy=update_rows, interpolation=cv2.INTER_LINEAR)
            
            # segmentation update: seg(x) * update_rows, seg(y) * update_rows
            new_seg = list()
            for index, seg in enumerate(json_data['annotations'][json_file_dic[f]]['segmentation'][0][0]):
                if ((index % 2) == 0):
                    new_seg.append(round(seg * update_cols, 1))
                else:
                    new_seg.append(round(seg * update_rows, 1))           
            
            # 저장할 파일 += 1
            save_file_name = str(int(save_file_name)+ 1)
            # 파일이 가위일 경우
            if "xray_scissors" in img_name:
                save_file_name = save_file_name.zfill(5)

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
            
            # 이미지 파일 저장
            cv2.imwrite(file_path + '/' + save_file_name + '.png', update_img)

    # json 파일 저장
    json_dump(json_path, json_data)

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


            

def filter_image(img_name):
    file_path = f"D:/wp/data/{img_name}/crop"
    json_path = f"D:/wp/data/{img_name}/json/crop_data.json"
    origin_img_path = f"D:/wp/data/원본/{img_name}/crop"
    add_img_path = f"D:/wp/data/{img_name}/add_image"

    origin_file = png_load(origin_img_path)
    file = png_load(file_path)
    
    json_data, json_file_dic = json_load(json_path)

    # origin_file에 없는 file 값들 저장
    s = set(origin_file)
    file = [x for x in file if x not in s]
    
    for i in file:
        original_img = cv2.imread(file_path + '/' + i, 1)
        filter_img = original_img.copy()
        
        # segmentation 불러오기
        seg = []
        for index, _ in enumerate(json_data['annotations'][json_file_dic[i]]['segmentation'][0][0]):
            if((index % 2) == 0):
                seg.append(([json_data['annotations'][json_file_dic[i]]['segmentation'][0][0]
                           [index], json_data['annotations'][json_file_dic[i]]['segmentation'][0][0][index+1]]))
        seg = np.array(seg, np.int32)

        # bbox 불러오기
        bbox = json_data['annotations'][json_file_dic[i]]['bbox']

        # seg 칠하기
        cv2.fillPoly(filter_img, [seg], (0, 0, 255))

        # bbox 그리기
        cv2.rectangle(filter_img, (bbox[0], bbox[1]),
                      (bbox[2], bbox[3]), (255, 0, 0, 1), 3)

        # original_img랑 filter_img 합성하기
        add_img = cv2.addWeighted(original_img, 0.7, filter_img, 0.3, 3)

        # add_image 폴더 없을 시 생성
        if not os.path.exists(add_img_path):
            os.makedirs(add_img_path)

        # 합성한 이미지 저장
        cv2.imwrite(add_img_path + '/' + i, add_img)

if __name__ == "__main__":
    main()