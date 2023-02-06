from library.utils.header import *
from library.utils.io import io
from library.utils.algorithm import algorithm

class groundtruths:
    def __init__(self, origin_img_path, json_path, ground_truths_path):
        self.origin_img_path = origin_img_path
        self.json_path = json_path
        self.ground_truths_path = ground_truths_path

    def run(self):

        origin_files = io.files_io.filesLoad(self.origin_img_path)

        json_data = io.json_io.jsonLoad(self.json_path)
        
        annotations = json_data['annotations']
        
        # 이전 이미지 이름 저장
        before_image_name = 0
        category_color = [(255, 0, 0), (0, 0, 255), (255, 255, 0), (0, 255, 0)]

        #annotaion 불러오기
        for annotation in annotations:
            image_id = annotation["image_id"]
            category_id = (annotation["category_id"])
            image_name = origin_files[image_id]

            if(before_image_name != image_name):
                original_img = cv2.imread(f"{self.origin_img_path}/{image_name}", 1)
                ground_truths_img = original_img.copy()
                before_image_name = image_name
            
            seg = algorithm.segTo2D(annotation)

            # bbox 불러오기
            bbox = annotation['bbox']

            # seg 칠하기
            cv2.fillPoly(ground_truths_img, [seg], category_color[category_id-1])

            # bbox 그리기
            cv2.rectangle(ground_truths_img, (bbox[0], bbox[1]),
                      (bbox[2]+bbox[0], bbox[3]+bbox[1]), category_color[category_id-1], 3)

            # original_img랑 filter_img 합성하기
            add_img = cv2.addWeighted(original_img, 0.7, ground_truths_img, 0.3, 3)

            # add_image 폴더 없을 시 생성
            if not os.path.exists(self.ground_truths_path):
                os.makedirs(self.ground_truths_path)

            # 합성한 이미지 저장
            if(before_image_name == image_name):
                io.image_io.imageSave(f"{self.ground_truths_path}/{image_name}", add_img)