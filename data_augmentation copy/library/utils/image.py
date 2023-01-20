from library.utils.header import cv2

class image:
    '''image load, save'''

    def load_image(image_path):
        '''Image load'''
        return cv2.imread(image_path, 1)

    def save_image(image_path, image_name, image):
        '''Image save'''
        cv2.imwrite(f'{image_path}/{image_name}.png', image)