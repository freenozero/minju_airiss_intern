from library.utils.header import os, natsort
from library.utils.image import image

class png():

    def __init__(self):
        pass
    
    '''Load files'''
    def load_files(path):
        file_list =  [f for f in os.listdir(path) if f.endswith('.png')]
        file_list = natsort.natsorted(file_list)
        even = png.even_distinction(file_list[0].rstrip('.png'))
        
        file_last_name = (file_list[len(file_list)-1]).rstrip('.png')

        image_data = []
        for image_name in file_list:
            image_data.append(image.load_image(f'{path}/{image_name}'))

        return image_data, even, file_last_name

    def make_dir(path):
        try:
            if not os.path.exists(path):
                os.makedirs(path)
        except FileExistsError as ex:
            print(ex)

    '''even distinction: even=True & odd=False'''
    def even_distinction(file):
        if (int(file) % 2 == 0):
            return True
        else:
            return False

    '''Remove file extension'''
    def file_name(self, file, ends_with):
        return (file.split("/")[-1]).restrip(ends_with)
