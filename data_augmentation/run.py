from augmentation.augmentation import augmentation

if __name__ == '__main__':
    aug = augmentation()
    path = "D:/wp/data/data_augmentation/xray_laserpointer_f_1"
    loop = 1
    image_data, even, file_last_name, json_data, json_file_dic = aug.set_data(path)
    resize_image_datas, groundtruths_json, json_data = aug.resize(path, image_data, json_data, even, file_last_name, json_file_dic, loop)
    groundtruths_data = aug.groundtruths(resize_image_datas, groundtruths_json)
    aug.save(path, groundtruths_data, resize_image_datas, json_data, file_last_name)