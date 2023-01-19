from library.utils.header import json as js
from library.utils.header import np

class json:

    '''Load json file'''
    def load_json(json_path):
        with open(json_path) as json_file:
            json_data = js.load(json_file)
        return json_data
    
    '''Dump json data'''
    def dump_json(json_path, json_data):
        with open(json_path, 'w') as json_file:
            js.dump(json_data, json_file)

    '''segmentation array return'''
    def return_segmentation(segmentation):
        result = np.array(segmentation[0][0])
        return result.reshape(int(len(result)/2), 2)

    '''Update segmentation'''
    def update_segmentation(segmentation, fy, fx):
        result = []
        for seg in segmentation:
            seg = np.array(seg)
            seg[::2] = seg[::2] * fx
            seg[1::2] = seg[1::2] * fy
            result.append(seg.tolist())
        return result