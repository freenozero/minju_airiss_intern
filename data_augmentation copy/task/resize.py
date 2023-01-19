# from library.images.groundtruths import Groundtruth

from library.utils.png import png
from library.utils.json import json

from task._abstract_ import AbstractTaskCase
'''Augment all the files in the folder'''
class augmentation(AbstractTaskCase):
    def __init__(self):
        pass
    #     self.gt = Groundtruth()

    def set_folder_data(self, path):
        image_data = png.load_files(f'{path}/crop')
        json_data, json_file_dic = json.load_json(f'{path}/json/crop_data.json')
        return image_data, json_data, json_file_dic
    
    def run(self):
        path = "D:/wp/data/data_augmentation/xray_scissors_5"
        image_data, json_data, json_file_dic = self.set_folder_data(path)
        print(image_data)

'''main'''
def run():
    aug = augmentation()
    aug.run()