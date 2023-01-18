from task._abstract_ import AbstractTaskCase

from library.images.groundtruths import Groundtruth

'''Augment all the files in the folder'''
class augmentation(AbstractTaskCase):
    def __init__(self):
        self.gt = Groundtruth()

    def set_folder_data(self, root):
        image_files = 


'''main'''
def run():
    aug = augmentation()
    aug.run()