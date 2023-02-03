from library.utils.header import *
from library.utils.io import io
from library.utils.algorithm import algorithm

class split:
    '''highlow 데이터를 high, low로 split'''
    def __init__(self, path):
        self.dataset_setting = ["train", "val", "test"]
        self.highlow_path=f'{path}/highlow'
        self.high_path =f'{path}/high'
        self.low_path = f'{path}/low'

    def dataSplit(self, highlow_jitter_files, highlow_json_data, high_json, low_json, dataset):
        high_file_name = 0
        low_file_name = 0
        for file_name in highlow_jitter_files:
            file_name = algorithm.delExtension(file_name, ".png")
            if(algorithm.even(file_name)):
                images = highlow_json_data["images"][file_name]
                save_jitter_path = f"{self.high_path}/{dataset}/jitter_image/{high_file_name}.png"

                # image 저장
                io.image_io.imageSave(save_jitter_path, io.image_io.imageLoad(images["path"]))

                images["id"] = high_file_name
                images["file_name"] = f"{high_file_name}.png"
                images["path"] = save_jitter_path
                high_json["images"].append(images)


                high_json = self.appendAnn(highlow_json_data, high_json, file_name, high_file_name)
                high_file_name += 1
            else:
                images = highlow_json_data["images"][file_name]
                save_jitter_path = f"{self.low_path}/{dataset}/jitter_image/{low_file_name}.png"

                # image 저장
                io.image_io.imageSave(save_jitter_path, io.image_io.imageLoad(images["path"]))

                images["id"] = low_file_name
                images["file_name"] = f"{low_file_name}.png"
                images["path"] = save_jitter_path
                low_json["images"].append(images)

                low_json = self.appendAnn(highlow_json_data, low_json, file_name, low_file_name)
                low_file_name += 1
        
        return high_json, low_json

    def appendAnn(self, highlow_json_data, save_json_data, file_name, save_file_name):
        '''annotation append'''
        for annotation in highlow_json_data["annotations"]:
            if(annotation["image_id"] == file_name):
                annotation["id"] = len(save_json_data["annotations"])
                annotation["image_id"] = save_file_name
                save_json_data["annotations"].append(annotation)
        return save_json_data

    def run(self):
        for dataset in self.dataset_setting:
            save_high_json_path = f"{self.high_path}/{dataset}/json/data.json"
            save_low_json_path = f"{self.low_path}/{dataset}/json/data.json"

            high_json = io.json_io.jsonLoad(save_high_json_path)
            low_json = io.json_io.jsonLoad(save_low_json_path)

            highlow_jitter_files = io.files_io.filesLoad(f"{self.highlow_path}/{dataset}/jitter_image")
            highlow_json_data = io.json_io.jsonLoad(f"{self.highlow_path}/{dataset}/json/data.json")
            high_json, low_json = self.dataSplit(highlow_jitter_files, highlow_json_data, high_json, low_json, dataset)

            io.json_io.jsonSave(high_json, save_high_json_path)
            io.json_io.jsonSave(low_json, save_low_json_path)

            high_json.clear()
            low_json.clear()