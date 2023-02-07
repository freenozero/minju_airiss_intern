from library.utils.header import *
from library.utils.io import io

if __name__ == "__main__":
    path = "D:/wp/data/background_image"
    files = io.files_io.filesLoad(path)
    for file in files:
        image = io.image_io.imageLoad(f"{path}/{file}")
        dst_image = cv2.resize(image, dsize=(700, 700), interpolation=cv2.INTER_AREA)
        io.image_io.imageSave(f"{path}/resize/{file}", dst_image)