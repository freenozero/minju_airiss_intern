from library.utils.header import *
from library.utils.io import io

class algorithm:
    def segTo2D(annotation):
        seg = []
        for index, x in enumerate(annotation['segmentation'][0]):
            if((index % 2) == 0):
                seg.append([x, annotation['segmentation'][0][index+1]])
        return(np.array(seg, np.int32))

    def delExtension(file_name, ext):
        '''file name extension delete and replace int'''
        return int(file_name.rstrip(ext))

    def even(file_name):
        '''if filename is even than return true'''
        if((file_name%2)==0):
            return True
        
    def mkDirFolder(path, highlow_setting, dataset_setting):
        '''make directory and folder for run.py
            data
                ㄴ manipulation_image
                    ㄴ highlow
                        ㄴ train
                            ㄴ image
                            ㄴ jitter_image
                            ㄴ ground_truth
                            ㄴ json
                                ㄴ data.json
                        ㄴ val
                            ㄴ image
                            ㄴ jitter_image
                            ㄴ ground_truth
                            ㄴ json
                                ㄴ data.json
                        ㄴ test
                            ㄴ image
                            ㄴ jitter_image
                            ㄴ ground_truth
                            ㄴ json
                                ㄴ data.json  
                    ㄴ high
                        ㄴ train
                            ㄴ image
                            ㄴ jitter_image
                            ㄴ ground_truth
                            ㄴ json       
                                ㄴ data.json
                        ㄴ val
                            ㄴ image
                            ㄴ jitter_image
                            ㄴ ground_truth
                            ㄴ json       
                                ㄴ data.json
                        ㄴ test
                            ㄴ image
                            ㄴ jitter_image
                            ㄴ ground_truth
                            ㄴ json       
                                ㄴ data.json
                    ㄴ low
                        ㄴ train
                            ㄴ image
                            ㄴ jitter_image
                            ㄴ ground_truth
                            ㄴ json       
                                ㄴ data.json
                        ㄴ val
                            ㄴ image
                            ㄴ jitter_image
                            ㄴ ground_truth
                            ㄴ json      
                                ㄴ data.json 
                        ㄴ test
                            ㄴ image
                            ㄴ jitter_image
                            ㄴ ground_truth
                            ㄴ json      
                                ㄴ data.json
        '''
        folder_setting = ["image", "jitter_image", "ground_truths", "json"]
        for highlow in highlow_setting:
            for dataset in dataset_setting:
                for folder_name in folder_setting:
                    mk_folder_path = f"{path}/{highlow}/{dataset}/{folder_name}"
                    algorithm.mkFolder(mk_folder_path)

                    if folder_name == "json":
                        algorithm.mkJsonFile(f"{path}/{highlow}/{dataset}/{folder_name}/data.json")
        return 0
    
    def mkFolder(path):
        if os.path.exists(path):
           for file in os.scandir(path):
               os.remove(file.path)
        else:
            os.makedirs(path)
    
    def mkJsonFile(path):
        '''make data.json file'''
        json_data = {'images':[], 'annotations':[], 'categories':
                                                            [{'id': 1,
                                                            'name': 'knife',
                                                            'supercategory': 'item',
                                                            'color': '040439',
                                                            'metadata': ''},
                                                            {'id': 2,
                                                            'name': 'gun',
                                                            'supercategory': 'item',
                                                            'color': '040439',
                                                            'metadata': ''},
                                                            {'id': 3,
                                                            'name': 'battery',
                                                            'supercategory': 'item',
                                                            'color': '040439',
                                                            'metadata': ''},
                                                            {'id': 4,
                                                            'name': 'laserpointer',
                                                            'supercategory': 'item',
                                                            'color': '040439',
                                                            'metadata': ''}]   }
        io.json_io.jsonSave(json_data, path)