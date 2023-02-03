import os
import cv2
import natsort

class inout:
    def fileLoad(file_path):
        file = [f for f in os.listdir(file_path)]
        file = natsort.natsorted(file)
        return file

    def imageLoad(image_path):
        return cv2.imread(image_path)

    def imageSave(image_path, image):
        cv2.imwrite(image_path, image)