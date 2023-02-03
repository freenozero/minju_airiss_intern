from library.utils.header import *
from library.utils.io import io

class jitter:
    def __init__(self, path):
        self.img_path = f"{path}/image"
        self.save_path = f"{path}/jitter_image"

    def run(self):
        file = io.files_io.filesLoad(self.img_path)
        color_aug = torchvision.transforms.ColorJitter(
                brightness=0.5, contrast=0.5, saturation=0.5)

        for file_name in file:
            img = io.image_io.imageLoadGrayScale(f"{self.img_path}\{file_name}")
            img = Image.fromarray(img)
            jitter_img = color_aug(img)
            jitter_img = cv2.cvtColor(np.array(jitter_img), cv2.COLOR_BGR2RGB)

            io.image_io.imageSave(f"{self.save_path}\{file_name}", jitter_img)