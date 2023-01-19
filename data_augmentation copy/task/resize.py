# from library.images.groundtruths import Groundtruth
from library.utils.header import random, cv2, os

from library.utils.png import png
from library.utils.json import json
from library.utils.image import image

from task._abstract_ import AbstractTaskCase
'''Augment all the files in the folder'''
class augmentation(AbstractTaskCase):

    def set_data(self, path):
        image_data, even, file_last_name = png.load_files(f'{path}/crop')
        json_data = json.load_json(f'{path}/json/crop_data.json')
        return image_data, even, file_last_name, json_data
    
    def resize(self, path, image_data, json_data, even, file_last_name, loop):
        image_data_copy = image_data.copy()
        for _ in range(0, loop):
            for index, original_image in enumerate(image_data_copy):
                file_last_name = f'{file_last_name[:-1]}{int(file_last_name[-1])+1}'
                
                if(even and png.even_distinction(file_last_name)):
                    update_cols = round(random.uniform(0.5, 1.5), 1)
                    update_rows = round(random.uniform(0.5, 1.5), 1)
                elif(not even and png.even_distinction(file_last_name)):
                    update_cols = round(random.uniform(0.5, 1.5), 1)
                    update_rows = round(random.uniform(0.5, 1.5), 1)

                update_image = cv2.resize(original_image, dsize=(
                0, 0), fx=update_cols, fy=update_rows, interpolation=cv2.INTER_LINEAR)

                seg = json.update_segmentation(json_data['annotations'][index]['segmentation'], update_rows, update_cols)
                
                new_images = {'id': len(json_data['images']) +1,
                            'dataset_id': json_data['images'][index]['dataset_id'],
                            'path': f'{path}/crop/{file_last_name}.png',
                            'file_name': file_last_name + '.png',
                            'width': update_image.shape[1],
                            'height': update_image.shape[0]}
                
                new_annotations = {'id': len(json_data['images']) +1,
                                    'image_id': len(json_data['images']) +1,
                                    'category_id': json_data['annotations'][index]['category_id'],
                                    # x, y, width, height
                                    'bbox': [0, 0, update_image.shape[1], update_image.shape[0]],
                                    'segmentation': seg,
                                    # height * width
                                    'area': update_image.shape[0]*update_image.shape[1],
                                    'iscrowd': json_data['annotations'][index]['iscrowd'],
                                    'color': json_data['annotations'][index]['color'],
                                    'unitID': json_data['annotations'][index]['unitID'],
                                    'registNum': json_data['annotations'][index]['registNum'],
                                    'number1': json_data['annotations'][index]['number1'],
                                    'number2': json_data['annotations'][index]['number2'],
                                    'weight': json_data['annotations'][index]['weight']}
                json_data['images'].append(new_images)
                json_data['annotations'].append(new_annotations)
                image_data.append(update_image)
        return image_data, json_data

    def groundtruths(self, image_data, json_data):
        groundtruths_data = []
        for index, original_image in enumerate(image_data):
            groundtruths_image = original_image.copy()
            seg = json.return_segmentation(json_data['annotations'][index]['segmentation'])
            bbox = json_data['annotations'][index]['bbox']
            cv2.fillConvexPoly(groundtruths_image, seg, (0, 0, 255))
            cv2.rectangle(groundtruths_image, (bbox[0], bbox[1]),
                                                (bbox[2], bbox[3]), (255, 0, 0), 3)
            
            groundtruths_image = cv2.addWeighted(original_image, 0.7, groundtruths_image, 0.3, 3)
            groundtruths_data.append(groundtruths_image)
        return groundtruths_data

    def save(self, path, resize_data, groundtruths_data, json_data, file_last_name):
            if not os.path.exists(f'{path}/groundtruths'):
                os.makedirs(f'{path}/groundtruths')

            for data in resize_data:
                file_last_name = f'{file_last_name[:-1]}{int(file_last_name[-1])+1}'
                image.save_image(f'{path}/crop', file_last_name, data)

            for data in groundtruths_data:
                file_last_name = f'{file_last_name[:-1]}{int(file_last_name[-1])+1}'
                image.save_image(f'{path}/groundtruths', file_last_name, data)

            json.dump_json(f'{path}/json/crop_data.json', json_data)