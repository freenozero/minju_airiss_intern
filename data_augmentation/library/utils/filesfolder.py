from library.utils.header import os, natsort

from library.utils.image import image

class filesfolder:

    def load_files(path):
        '''Load png files'''
        file_list =  [f for f in os.listdir(path) if f.endswith('.png')]
        file_list = natsort.natsorted(file_list)
        even = filesfolder.even_distinction(file_list[0].rstrip('.png'))
        
        file_last_name = (file_list[len(file_list)-1]).rstrip('.png')

        image_data = []
        for image_name in file_list:
            image_data.append(image.load_image(f'{path}/{image_name}'))

        return image_data, even, file_last_name

    ''' file_name += 1 and scissors는 이름 변환'''
    def file_naming(path, file_last_name):
        file_last_name = str(int(file_last_name)+ 1)
        if "xray_scissors" in path:
            file_last_name = file_last_name.zfill(5)
        return file_last_name

    def even_distinction(file):
        '''even distinction: even=True & odd=False'''
        if (int(file) % 2 == 0):
            return True
        else:
            return False
    
    def make_dir(path):
        '''make directory'''
        try:
            if not os.path.exists(path):
                os.makedirs(path)
        except FileExistsError as ex:
            print(ex)

    def file_name(self, file, ends_with):
        '''Remove file extension'''
        return (file.split("/")[-1]).restrip(ends_with)
