from library.utils.header import cv2

class image:
    '''Image load'''
    def load_image(image_path):
        return cv2.imread(image_path, 1)

    '''Image save'''
    def save_image(image_path, image_name, image):
        cv2.imwrite(f'{image_path}/{image_name}', image)