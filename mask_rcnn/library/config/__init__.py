import importlib
import os
import sys

from library.utils.io import split_path_name, check_dir_list

def config_set(config_full_path):
    config_path, config_name = split_path_name(file_path_name=config_full_path)

    module_name = os.path.splitext(config_name)[0]

    sys.path.insert(0, config_path)

    load_module = importlib.import_module(module_name)

    configs = {
        name : value
        for name, value in load_module.__dict__.items()
        if not name.startswith("__")
    }
    
    if "dirs" in configs.keys():
        check_dir_list(configs["dirs"])
    return configs

train_configs = config_set("./mask_rcnn/library/config/train_config.py")
predict_configs = config_set("./mask_rcnn/library/config/predict_config.py")
view_configs = config_set("./library/ai/MaskRCNN/config/view_config.py")