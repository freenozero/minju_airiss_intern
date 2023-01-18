from library.utils.header import os, natsort

class png():

    def __init__(self, ends_with):
        self.ends_with = ends_with
        self.even = True
    
    '''Load files'''
    def load_files(self, path):
        file_list = [f for f in os.listdir(path) if f.endswith(self.ends_with)]
        return natsort.natsorted(file_list)

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
    def file_name(self, file):
        return (file.split("/")[-1]).restrip(self.ends_with)
