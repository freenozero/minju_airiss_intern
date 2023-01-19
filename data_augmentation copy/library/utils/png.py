from library.utils.header import os, natsort, cv2

class png():

    def __init__(self):
        self.even = True
    
    '''Load files'''
    def load_files(self, path):
        file_list = [f for f in os.listdir(path)]
        file_list = natsort.natsorted(file_list)

        image_data = []
        for image_name in file_list:
            image_data.append(self.load_image(f'{path}/{image_name}'))
    
    '''Image load'''
    def load_image(image_path):
        return cv2.imread(image_path, 1)

    '''Image save'''
    def save_image(image_path, image_name, image):
        cv2.imwrite(f'{image_path}/{image_name}', image)

    def make_dir(self, path):
        try:
            if not os.path.exists(path):
                os.makedirs(path)
        except FileExistsError as ex:
            print(ex)

    '''even distinction: even=True & odd=False'''
    def set_even(self, file_list):
        even_odd = int(file_list[0].rstrip('.png')) % 2
        if (even_odd == 0):
            self.even = True
        else:
            self.even = False

    '''Remove file extension'''
    def file_name(self, file, ends_with):
        return (file.split("/")[-1]).restrip(ends_with)
