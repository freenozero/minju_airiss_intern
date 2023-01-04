import json
import cv2
import numpy as np

json_path = 'D:/wp/data/xray_artknife_a_1/json/crop_data.json'
with open(json_path) as json_file:
    json_data = json.load(json_file)

for i in range(52, 77):
    img = cv2.imread(json_data['images'][i]['path'], -1)
    cv2.waitKey(0)
    print(i)
    seg = []
    for j in range(0, len(json_data['annotations'][i]['segmentation'][0][0])):
        if j % 2 == 0:
            seg.append(([json_data['annotations'][i]['segmentation'][0][0][j], json_data['annotations'][i]['segmentation'][0][0][j+1]]))

    seg = np.array(seg, np.int32)
    seg_img = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    cv2.fillConvexPoly(seg_img, seg, color=(255, 255, 0))
    
    cv2.imshow("seg_img", seg_img)
    cv2.waitKey(0)