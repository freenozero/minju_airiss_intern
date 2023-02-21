import json
import os
import cv2
import numpy as np
import random
import natsort

# json에 category_id는 직접 변경
def main():
    #knife(o)
    #gun(o)
    #battery(o)
    #laserpointer(o)
    img_name = "knife"
    folder_path = f"D:/wp/data/background_manipulation/augmentation&cateogrical/before_aug/{img_name}"
    folder_name = os.listdir(folder_path)
    augmentation(img_name, folder_name)
    # filter_image(img_name)

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

# 이미지 증강
def augmentation(img_name, folder_name):

    save_json_data = {'images':[], 'annotations':[], 'categories':
                                                        [{'id': 0,
                                                        'name': 'knife',
                                                        'supercategory': 'item',
                                                        'color': '040439',
                                                        'metadata': ''},
                                                        {'id': 1,
                                                        'name': 'gun',
                                                        'supercategory': 'item',
                                                        'color': '040439',
                                                        'metadata': ''},
                                                        {'id': 2,
                                                        'name': 'battery',
                                                        'supercategory': 'item',
                                                        'color': '040439',
                                                        'metadata': ''},
                                                        {'id': 3,
                                                        'name': 'laserpointer',
                                                        'supercategory': 'item',
                                                        'color': '040439',
                                                        'metadata': ''}]   }

    # 저장할 파일 이름은 0부터 9999
    save_file_name = 0
    # low, high 이미지 크기를 같게 하기 위함
    img_size = tuple()

    while(save_file_name < 9999):
        
        # 카테고리
        for i in folder_name:
            if save_file_name > 9999:
                break

            file_path = f"D:/wp/data/background_manipulation/augmentation&cateogrical/before_aug/{img_name}"
            json_path = f"{file_path}/{i}/json/crop_data.json"

            save_path = f"D:/wp/data/background_manipulation/augmentation&cateogrical/after_aug/{img_name}"
            save_json_path = f"{save_path}/json/crop_data.json"

            file_path = f"{file_path}/{i}/crop"
            save_path = f"{save_path}/crop"
            
            file = png_load(file_path)
            json_data, json_file_dic = json_load(json_path)

            # 시작 파일 짝수 홀수
            first_file_name = int(file[0].rstrip('.png')) % 2

            # 파일 하나씩 불러오기
            for f in file:
                if save_file_name > 9999:
                    break
                
                file_name = int(f.rstrip('.png'))

                # -1은 16 bit 이미지 불러오기 위함
                img = cv2.imread(file_path + '/' + f, -1)

                if (first_file_name == 0 and file_name % 2 == 0): #시작 파일이 짝수고, 현재 증강할 파일이 짝수면 random
                        update_cols = round(random.uniform(0.5, 1.5), 1)
                        update_rows = round(random.uniform(0.5, 1.5), 1)
                        # 상대적인 비율로 축소, 확대
                        img = cv2.resize(img, dsize=(
                            0, 0), fx=update_cols, fy=update_rows, interpolation=cv2.INTER_LINEAR)
                        img_size = (img.shape[1], img.shape[0])
                elif (first_file_name != 0 and file_name % 2 != 0): #시작 파일이 홀수고, 현재 증강할 파일이 홀수면 random
                        update_cols = round(random.uniform(0.5, 1.5), 1)
                        update_rows = round(random.uniform(0.5, 1.5), 1)
                        # 상대적인 비율로 축소, 확대
                        img = cv2.resize(img, dsize=(
                            0, 0), fx=update_cols, fy=update_rows, interpolation=cv2.INTER_LINEAR)
                        img_size = (img.shape[1], img.shape[0])
                else:
                    img = cv2.resize(img, img_size, interpolation=cv2.INTER_LINEAR)
                

                # segmentation update: seg(x) * update_rows, seg(y) * update_rows
                new_seg = list()
                for index, seg in enumerate(json_data['annotations'][json_file_dic[f]]['segmentation'][0][0]):
                    if ((index % 2) == 0):
                        new_seg.append(round(seg * update_cols, 1))
                    else:
                        new_seg.append(round(seg * update_rows, 1))           

                # json 파일 추가
                new_images = {'id': save_file_name,
                                'dataset_id': json_data['images'][json_file_dic[f]]['dataset_id'],
                                'path': save_path + '/' + str(save_file_name) + '.png',
                                'file_name': str(save_file_name) + '.png',
                                'width': img.shape[1],
                                'height': img.shape[0]}
                save_json_data['images'].append(new_images)

                new_annotations = {'id': save_file_name+1,
                                    'image_id': save_file_name+1,
                                    # ※1: knife, 2:gun, 3:battery, 4:laserpointer 변경 필요※
                                    'category_id': 1,
                                    # x, y, width, height
                                    'bbox': [0, 0, img.shape[1], img.shape[0]],
                                    'segmentation': [[new_seg]],
                                    # height * width
                                    'area': img.shape[0]*img.shape[1],
                                    'iscrowd': False,
                                    'color': 'Unknown',
                                    'unitID': 1,
                                    'registNum': 1,
                                    'number1': 4,
                                    'number2': 4,
                                    'weight': None}
                save_json_data['annotations'].append(new_annotations)
                # 이미지 파일 저장
                cv2.imwrite(save_path + '/' + str(save_file_name) + '.png', img)
                save_file_name += 1

            # json 파일 저장
            json_dump(save_json_path, save_json_data)

# 이미지 증강 잘 되었는지 확인
def filter_image(img_name):    
    file_path = f"D:/wp/data/background_manipulation/augmentation&cateogrical/after_aug/{img_name}/crop"
    json_path = f"D:/wp/data/background_manipulation/augmentation&cateogrical/after_aug/{img_name}/json/crop_data.json"
    add_img_path = f"D:/wp/data/background_manipulation/augmentation&cateogrical/after_aug/{img_name}/filter_image"

    file = png_load(file_path)

    json_data, json_file_dic = json_load(json_path)
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