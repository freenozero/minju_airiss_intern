from library.utils.header import cv2, os, np

from library.utils.image import image
from library.utils.json import json
from library.utils.filesfolder import filesfolder
from library.utils.reshape import reshape

from augmentation._abstract_ import AbstractTaskCase

class augmentation(AbstractTaskCase):
    '''Augment all the files in the folder'''
    
    def set_data(self, path):
        image_data, even, file_last_name = filesfolder.load_files(f'{path}/crop')
        json_data, json_file_dic = json.load_json(f'{path}/json/crop_data.json')
        return image_data, even, file_last_name, json_data, json_file_dic
    
    def resize(self, path, image_data, json_data, even, file_last_name, json_file_dic, loop):
        image_data_copy = image_data.copy()
        update_cols = 1
        update_rows = 1
        resize_image_datas = []
        groundtruths_json = []

        file_num = (int(file_last_name) - len(image_data)+1)
        for _ in range(0, loop):
            for index, original_image in enumerate(image_data_copy):
                file_last_name = filesfolder.file_naming(path, file_last_name)

                update_cols, update_rows = reshape.random(even, file_last_name, update_cols, update_rows)              

                update_image = cv2.resize(original_image, dsize=(
                0, 0), fx=update_cols, fy=update_rows, interpolation=cv2.INTER_LINEAR)

                seg = reshape.update_segmentation(json_data['annotations'][json_file_dic[f'{index}.png']]['segmentation'][0][0], update_cols, update_rows)
                
                new_images = {'id': len(json_data['images']) +1,
                            'dataset_id': json_data['images'][json_file_dic[f'{index}.png']]['dataset_id'],
                            'path': f'{path}/crop/{file_last_name}.png',
                            'file_name': f'{file_last_name}.png',
                            'width': update_image.shape[1],
                            'height': update_image.shape[0]}
                
                new_annotations = {'id': len(json_data['images']) +1,
                                    'image_id': len(json_data['images']) +1,
                                    'category_id': json_data['annotations'][json_file_dic[f'{index}.png']]['category_id'],
                                    # x, y, width, height
                                    'bbox': [0, 0, update_image.shape[1], update_image.shape[0]],
                                    'segmentation': seg,
                                    # height * width
                                    'area': update_image.shape[0]*update_image.shape[1],
                                    'iscrowd': json_data['annotations'][json_file_dic[f'{index}.png']]['iscrowd'],
                                    'color': json_data['annotations'][json_file_dic[f'{index}.png']]['color'],
                                    'unitID': json_data['annotations'][json_file_dic[f'{index}.png']]['unitID'],
                                    'registNum': json_data['annotations'][json_file_dic[f'{index}.png']]['registNum'],
                                    'number1': json_data['annotations'][json_file_dic[f'{index}.png']]['number1'],
                                    'number2': json_data['annotations'][json_file_dic[f'{index}.png']]['number2'],
                                    'weight': json_data['annotations'][json_file_dic[f'{index}.png']]['weight']}
                json_data['images'].append(new_images)
                json_data['annotations'].append(new_annotations)
                resize_image_datas.append(update_image)
                groundtruths_json.append([seg[0][0], new_annotations['bbox']])
        return resize_image_datas, groundtruths_json, json_data

    def groundtruths(self, resize_image_datas, groundtruths_json):
        groundtruths_data = []
        for index, original_image in enumerate(resize_image_datas):
            groundtruths_image = original_image.copy()
            seg, bbox = reshape.return_segmentation(groundtruths_json[index][0]), groundtruths_json[index][1]
            cv2.fillConvexPoly(groundtruths_image, seg, (0, 0, 255))
            cv2.rectangle(groundtruths_image, (bbox[0], bbox[1]),
                                                (bbox[2], bbox[3]), (255, 0, 0, 1), 3)
            
            groundtruths_image = cv2.addWeighted(original_image, 0.7, groundtruths_image, 0.3, 3)
            groundtruths_data.append(groundtruths_image)
        return groundtruths_data

    def save(self, path, groundtruths_data, resize_image_datas, json_data, file_last_name):
            if not os.path.exists(f'{path}/groundtruths'):
                os.makedirs(f'{path}/groundtruths')

            for i in range(len(groundtruths_data)):
                file_last_name = filesfolder.file_naming(path, file_last_name)
                image.save_image(f'{path}/crop', file_last_name, resize_image_datas[i])
                image.save_image(f'{path}/groundtruths', file_last_name, groundtruths_data[i])

            json.dump_json(f'{path}/json/crop_data.json', json_data)