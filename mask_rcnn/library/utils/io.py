from library.utils.header import *


def load_image(options):
    '''
    options = {
        "file_name": {file_name},
        "dtype": {dtype}
    }
    '''
    file_name = options["file_name"]
    dtype = options["dtype"]

    img = Image.open(file_name)
    img.load()
    data = np.asarray(img, dtype=dtype)
    return data


def save_image(data, options):
    '''
    options = {
        "file_name": {file_name},
        "dtype": {dtype},
        "start_pixel": {start_pixel},
        "end_pixel": {end_pixel}
    }
    '''
    file_name = options["file_name"]
    dtype = options["dtype"]
    start_pixel = options["start_pixel"]
    end_pixel = options["end_pixel"]

    img = Image.fromarray(
        np.asarray(
            np.clip(data, start_pixel, end_pixel), dtype=dtype
        )
    )
    img.save(file_name)
    

def cv_load_image(load_path, file_name, dtype):
    image_array = np.fromfile(f"{load_path}/{file_name}", np.uint8)
    result = cv2.imdecode(image_array, dtype)
    return result

def cv_save_image(data, save_path, file_name, dtype, start_pixel, end_pixel):

    extension = os.path.splitext(file_name)[1]
    
    data = np.asarray(
        np.clip(data, start_pixel, end_pixel), dtype=dtype
    )
    
    result, encoded_img = cv2.imencode(extension, data)
    
    if result: 
        with open(f"{save_path}/{file_name}", mode="w+b") as f: 
            encoded_img.tofile(f)
            

def load_json(load_path, file_name):
    with open(f"{load_path}/{file_name}", encoding="utf-8") as json_file:
        json_data = json.load(json_file)
    return json_data


def save_json(json_data, save_path, file_name):
    with open(f"{save_path}/{file_name}", "w", encoding="utf-8") as json_file:
        json.dump(json_data, json_file, ensure_ascii=False)

def split_path_name(file_path_name):
    path = "/".join(file_path_name.split("/")[:-1])
    name = file_path_name.split("/")[-1]
    return path, name

def split_extension_file_name(file_name):
    return "".join(file_name.split(".")[:-1])

def check_dir_list(dir_dict):
    for path in (list(dir_dict.values())):
        check_dir(path)

def check_dir(path):
    os.makedirs(path, exist_ok=True)
