from augmentation.augmentation import augmentation

if __name__ == '__main__':
    aug = augmentation()
    path = "D:/wp/data/data_augmentation/xray_scissors_5"
    loop = 2
    image_data, even, file_last_name, json_data = aug.set_data(path)
    image_data, json_data = aug.resize(path, image_data, json_data, even, file_last_name, loop)
    groundtruths_data = aug.groundtruths(image_data, json_data)
    aug.save(path, image_data, groundtruths_data, json_data, file_last_name)