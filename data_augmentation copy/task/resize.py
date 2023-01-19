# from library.images.groundtruths import Groundtruth
from library.utils.header import random, cv2, np

from library.utils.png import png
from library.utils.json import json
from library.utils.image import image

from task._abstract_ import AbstractTaskCase
'''Augment all the files in the folder'''
class augmentation(AbstractTaskCase):
    def __init__(self):
        pass
    #     self.gt = Groundtruth()

    def set_data(self, path):
        image_data, even, file_last_name = png.load_files(f'{path}/crop')
        json_data = json.load_json(f'{path}/json/crop_data.json')

        return image_data, even, file_last_name, json_data
    
    def resize(self, image_data, json_data, even, file_last_name, path):
        image_data_copy = image_data.copy()
        path = f'{path}/crop'
        for _ in range(0, 1):
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

                new_seg = list()
                for index, seg in enumerate(json_data['annotations'][index]['segmentation'][0][0]):
                    if ((index % 2) == 0):
                        new_seg.append(round(seg * update_cols, 1))
                    else:
                        new_seg.append(round(seg * update_rows, 1)) 
                
                new_images = {'id': len(json_data['images']) +1,
                            'dataset_id': json_data['images'][index]['dataset_id'],
                            'path': f'{path}/crop/{file_last_name}.png',
                            'file_name': file_last_name + '.png',
                            'width': update_image.shape[1],
                            'height': update_image.shape[0]}
                json_data['images'].append(new_images)
                new_annotations = {'id': len(json_data['images']),
                                    'image_id': len(json_data['images']),
                                    'category_id': json_data['annotations'][index]['category_id'],
                                    # x, y, width, height
                                    'bbox': [0, 0, update_image.shape[1], update_image.shape[0]],
                                    'segmentation': [[new_seg]],
                                    # height * width
                                    'area': update_image.shape[0]*update_image.shape[1],
                                    'iscrowd': json_data['annotations'][index]['iscrowd'],
                                    'color': json_data['annotations'][index]['color'],
                                    'unitID': json_data['annotations'][index]['unitID'],
                                    'registNum': json_data['annotations'][index]['registNum'],
                                    'number1': json_data['annotations'][index]['number1'],
                                    'number2': json_data['annotations'][index]['number2'],
                                    'weight': json_data['annotations'][index]['weight']}
                json_data['annotations'].append(new_annotations)
                image.save_image(f'{path}/crop', file_last_name, update_image)

        json.dump_json(f'{path}/json/crop_data.json', json_data)
        return image_data, json_data

    # def save(image_data, json_data):
    #     pass

    def view(image_data, json_data):
        pass

    def run(self):
        path = "D:/wp/data/data_augmentation/xray_scissors_5"
        image_data, even, file_last_name, json_data = self.set_data(path)
        image_data, json_data = self.resize(image_data, json_data, even, file_last_name, path)
        self.view(image_data, json_data)
        

'''main'''
def run():
    aug = augmentation()
    aug.run()