from library.utils.io import io
from library.utils.header import *

# json load
def jsonLoad():
    json_data = []
    json_path = ["D:/wp/data/categorical_image/knife/json/crop_data.json",
                "D:/wp/data/categorical_image/gun/json/crop_data.json",
                "D:/wp/data/categorical_image/bettery/json/crop_data.json",
                "D:/wp/data/categorical_image/laserpointer/json/crop_data.json"]

    for path in json_path:
        json_data.append(io.json_io.jsonLoad(path))
    return json_data

# json append
def jsonAppend(original_json, save_json, file_name, bk_images, random_max,  high_item, low_item, save_path):
    for index, image in enumerate(bk_images):
        # high
        if ((index % 2) == 0):
            item_images = allItemLoad(high_item)
            new_images = {'id': file_name,
                                'dataset_id': 1,
                                'path': save_path + '/' + str(file_name) + '.png',
                                'file_name': str(file_name) + '.png',
                                'width': image.shape[1],
                                'height': image.shape[0]}
            save_json['images'].append(new_images)
            for category, file_index in enumerate(high_item):
                for index, original_file_name in enumerate(file_index):
                    item_image = item_images[category][index]
                    new_seg = list()
                    xy = random_max[category][index]
                    # time.sleep(1)
                    for index, seg in enumerate(original_json[category]['annotations'][original_file_name]['segmentation'][0][0]):
                        if ((index % 2) == 0):
                            new_seg.append(seg+(xy[1]-item_image.shape[1]))
                        else:
                            new_seg.append(seg+(xy[0]-item_image.shape[0]))
                    annotations_len = len(save_json['annotations'])
                    new_annotations = {'id': annotations_len, # 0부터 증가
                                        'image_id': file_name,
                                        'category_id': category+1,
                                        # x, y, width, height
                                        # bk_image[random_max[j][z][0]-item_height:random_max[j][z][0], random_max[j][z][1]-item_width:random_max[j][z][1]] *= item_image
                                        # xy[0]: max_height
                                        # xy[1]: max_width
                                        # item_image.shape[0]: height
                                        # item_image.shape[1]: width
                                        'bbox': [xy[1]-item_image.shape[1], xy[0]-item_image.shape[0], item_image.shape[1], item_image.shape[0]],
                                        'segmentation': [new_seg],
                                        # height * width
                                        'area': (xy[1]-item_image.shape[1])*(xy[0]-item_image.shape[0])*(item_image.shape[0])*(item_image.shape[1]),
                                        'iscrowd': False,
                                        'color': 'Unknown',
                                        'unitID': 1,
                                        'registNum': 1,
                                        'number1': 4,
                                        'number2': 4,
                                        'weight': None}
                    save_json['annotations'].append(new_annotations)
        else:
            item_images = allItemLoad(low_item)
            new_images = {'id': file_name + 1,
                                'dataset_id': 1,
                                'path': save_path + '/' + str(file_name +1) + '.png',
                                'file_name': str(file_name +1) + '.png',
                                'width': bk_images[0].shape[1],
                                'height': bk_images[0].shape[0]}
            save_json['images'].append(new_images)
            for category, file_index in enumerate(low_item):
                for index, original_file_name in enumerate(file_index):
                    item_image = item_images[category][index]
                    new_seg = list()
                    # print(random_max[category][index])
                    xy = random_max[category][index]
                    # time.sleep(1)
                    for index, seg in enumerate(original_json[category]['annotations'][original_file_name]['segmentation'][0][0]):
                        if ((index % 2) == 0):
                            new_seg.append(seg+(xy[1]-item_image.shape[1]))
                        else:
                            new_seg.append(seg+(xy[0]-item_image.shape[0]))
                    annotations_len = len(save_json['annotations'])
                    new_annotations = {'id': annotations_len, # 0부터 증가
                                        'image_id': file_name + 1,
                                        'category_id': category+1,
                                        # x, y, width, height
                                        # bk_image[random_max[j][z][0]-item_height:random_max[j][z][0], random_max[j][z][1]-item_width:random_max[j][z][1]] *= item_image
                                        # xy[0]: max_height
                                        # xy[1]: max_width
                                        # item_image.shape[0]: height
                                        # item_image.shape[1]: width
                                        'bbox': [xy[1]-item_image.shape[1], xy[0]-item_image.shape[0], item_image.shape[1], item_image.shape[0]],
                                        'segmentation': [new_seg],
                                        # height * width
                                        'area': (xy[1]-item_image.shape[1])*(xy[0]-item_image.shape[0])*(item_image.shape[0])*(item_image.shape[1]),
                                        'iscrowd': False,
                                        'color': 'Unknown',
                                        'unitID': 1,
                                        'registNum': 1,
                                        'number1': 4,
                                        'number2': 4,
                                        'weight': None}
                    save_json['annotations'].append(new_annotations)
    return save_json

# item 모든 이미지 불러오기
def allItemLoad(item):
    #categorical
    categorical = ["knife", "gun", "bettery", "laserpointer"]
    item_images = [[],[],[],[]]
    # 모든 이미지 불러오기
    for i, category in enumerate(categorical):
        for file_name in item[i]:
            image_path = f"D:/wp/data/categorical_image/{category}/crop/{file_name}.png"
            item_images[i].append(io.image_io.imageLoad(image_path))
    return item_images

# 이미지 index 랜덤 추출
def itemIndexRandom(item_used):
    '''카테고리당 3개에서 5개 추출'''
    zero = 0
    category_image_max = 10000
    random_start = 3
    random_end = 5
    high_item = [[],[],[],[]]
    low_item = [[],[],[],[]]

    # 카테고리
    for i in range (zero, len(item_used)):
        random_cnt = random.randint(random_start, random_end)

        # low, high 합쳐서 합성할 해당 카테고리 갯수
        random_cnt *= 2

        # item_used는 다음부터 사용할 이미지까지의 이름을 가짐
        # item에는 합성할 이미지들을 저장함
        # random_cnt는 사용할 이미지의 총 갯수

        # 사용할 카테고리 len이 max를 넘으면 초기화.
        if((item_used[i]+random_cnt) >= category_image_max):
            item_used[i] = zero
        
        for index in range(item_used[i], (random_cnt + item_used[i]), 2):
            high_item[i].append(index)
            low_item[i].append(index+1)
        item_used[i] += random_cnt
    return item_used, high_item, low_item

def manipulation(bk_images, high_item, low_item):
    max_pixel = 65535 #16비트 max_pixel
    random_max = [[],[],[],[]]
    for i, bk_image in enumerate(bk_images):
        bk_image = (bk_image/max_pixel)

        # item all image 불러오기 [[],[],[],[]]
        # high
        if(i % 2 == 0):
            item_images = allItemLoad(high_item)
        # low
        else:
            item_images = allItemLoad(low_item)
            
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

                # low
                else:
                    bk_image[random_max[j][z][0]-item_height:random_max[j][z][0], random_max[j][z][1]-item_width:random_max[j][z][1]] *= item_image
        bk_images[i] = cv2.normalize(bk_image, None, 0, max_pixel, cv2.NORM_MINMAX, dtype=cv2.CV_16U)
    return bk_images, random_max

def run(setting):
    # path
    background_path = "D:/wp/data/background_image/resize"
    save_path = "D:/wp/data/manipulation_image/highlow"

    item_used = [0 for i in range(4)]

    # category json
    original_json = jsonLoad()

    # image file name load
    background_file = io.files_io.filesLoad(background_path)

    # train val test
    for set in setting:
        save_json_path = f"{save_path}/{set}/json/data.json"
        save_image_path = f"{save_path}/{set}/image"

        save_json = io.json_io.jsonLoad(save_json_path)

        file_name = 0
        while (file_name <= setting[set]):

            # background index
            bk_random_index = random.randrange(0, 8, 2)
            # background image list
            bk_images = []
            
            # background
            bk_images.append(io.image_io.imageLoad(f"{background_path}/{background_file[bk_random_index]}"))
            bk_images.append(io.image_io.imageLoad(f"{background_path}/{background_file[bk_random_index+1]}"))

            # itemImage
            item_used, high_item, low_item = itemIndexRandom(item_used)

            # image manipulation
            bk_images, random_max = manipulation(bk_images, high_item, low_item)
            
            # image save
            io.image_io.imageSave(file_path=f"{save_image_path}/{file_name}.png", image=bk_images[0])
            io.image_io.imageSave(file_path=f"{save_image_path}/{file_name+1}.png", image=bk_images[1])

            # json append(highlow)
            save_json = jsonAppend(original_json, save_json, file_name, bk_images, random_max, high_item, low_item, f"D:/wp/data/manipulation_image/highlow/{set}/jitter_image")


            file_name += 2

        # json save
        io.json_io.jsonSave(save_json, save_json_path)

if __name__ == "__main__":
    run()