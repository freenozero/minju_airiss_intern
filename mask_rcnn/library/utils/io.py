from library.utils.header import *

def split_path_name(file_path_name):
    path = "/".join(file_path_name.split("/")[:-1])
    name = file_path_name.split("/")[-1]
    return path, name

def check_dir_list(dir_dict):
    for path in (list(dir_dict.values())):
        check_dir(path)

def check_dir(path):
    os.makedirs(path, exist_ok=True)

def save_json(json_data, save_path, file_name):
    with open(f"{save_path}/{file_name}", "w", encoding="utf-8") as json_file:
        json.dump(json_data, json_file, ensure_ascii=False)
