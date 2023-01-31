import torchvision
import cv2
from PIL import Image
import numpy as np
from inout import inout

img_root=r'D:\wp\data\background_manipulation\manipulation\manipulation_image\image'
dst_root=r'D:\wp\data\background_manipulation\manipulation\manipulation_image\jitter_image'

if __name__ == '__main__':
    file = inout.fileLoad(img_root)
    color_aug = torchvision.transforms.ColorJitter(
            brightness=0.5, contrast=0.5, saturation=0.5)

    for file_name in file:
        img = inout.imageLoad(f"{img_root}\{file_name}")

        img = Image.fromarray(img)
        jitter_img = color_aug(img)
        jitter_img = cv2.cvtColor(np.array(jitter_img), cv2.COLOR_BGR2RGB)

        inout.imageSave(f"{dst_root}\{file_name}", jitter_img)