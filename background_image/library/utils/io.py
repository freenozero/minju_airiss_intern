from library.utils.header import *

class io:
    class files_io:
        '''file inout'''

        def filesLoad(file_path):
            '''png 파일만 정렬해서 return'''
            file = [f for f in os.listdir(file_path) if f.endswith('.png')]
            file = natsort.natsorted(file)
            return file

    class image_io:
        '''image inout'''

        def imageLoad(file_path):
            return cv2.imread(file_path, -1)

        def imageLoadGrayScale(file_path):
            return cv2.imread(file_path)

        def imageSave(file_path, image):
            cv2.imwrite(file_path, image)    

    class json_io:
        '''json inout'''

        def jsonLoad(file_path):
            with open(file_path) as json_file:
                json_data = json.load(json_file)
            return json_data

        def jsonSave(json_data, file_path):
            with open(file_path, 'w') as json_file:
                json_data = json.dump(json_data, json_file)
            return json_data