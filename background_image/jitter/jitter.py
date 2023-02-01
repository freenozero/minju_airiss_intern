import torchvision
import cv2
from PIL import Image
import numpy as np
from jitter.inout import inout

class jitter:
    def run(set):
        img_root=f'D:/wp/data/manipulation_jitter_image/{set}/image'
        dst_root=f'D:/wp/data/manipulation_jitter_image/{set}/jitter_image'
        file = inout.fileLoad(img_root)
        color_aug = torchvision.transforms.ColorJitter(
                brightness=0.5, contrast=0.5, saturation=0.5)

        for file_name in file:
            img = inout.imageLoad(f"{img_root}\{file_name}")

            img = Image.fromarray(img)
            jitter_img = color_aug(img)
            jitter_img = cv2.cvtColor(np.array(jitter_img), cv2.COLOR_BGR2RGB)

            inout.imageSave(f"{dst_root}\{file_name}", jitter_img)