# 주어진 test 이미지에서 사용하지 않는 json 내용 none으로 변경
from library.utils.header import *
from library.utils.io import io

# categories:
# id: 1 = knife
#     2 = gun<gasgun, toygun>
#     3 = bettray(철자 무엇..)
#     4 = laserpointer

# test categories:
# id: 1 = gun -> 2번으로 변경
#     2 = laserpointer -> 4번으로 변경
#     3 = tasergun -> none으로 변경
#     4 = magazine -> none으로 변경
#     5 = knife -> 1번으로 변경
#     6 = battery -> 3번으로 변경
#     7 = container -> none으로 변경
#     8 = taserbar -> none으로 변경
#     9 = gasgun -> 2번으로 변경
#     10 = fuse -> none으로 변경

json_path = "D:/wp/data/test_image/High_energy_image_size(700x700)_16bit/json"
json_data = io.json_io.jsonLoad(f"{json_path}/data.json")
save_json_data = {'images':[], 'annotations':[], 'categories':
                                                    [{'id': 1,
                                                    'name': 'knife',
                                                    'supercategory': 'item',
                                                    'color': '040439',
                                                    'metadata': ''},
                                                    {'id': 2,
                                                    'name': 'gun',
                                                    'supercategory': 'item',
                                                    'color': '040439',
                                                    'metadata': ''},
                                                    {'id': 3,
                                                    'name': 'battery',
                                                    'supercategory': 'item',
                                                    'color': '040439',
                                                    'metadata': ''},
                                                    {'id': 4,
                                                    'name': 'laserpointer',
                                                    'supercategory': 'item',
                                                    'color': '040439',
                                                    'metadata': ''}]   }
save_json_data["images"] = json_data["images"]

categories = json_data["categories"]
original_categories = {}
for cate in categories:
    original_categories.update({cate["name"]: cate["id"]})

change_categories = {"knife":1, "gun":2, "gasgun":2, "battery":3, "laserpointer":4}

for ann in json_data["annotations"]:
    for change_cate in change_categories:
        if ((list(original_categories.keys())[ann["category_id"]-1]) == change_cate):
            ann["category_id"] = change_categories[change_cate]
            save_json_data["annotations"].append(ann)
            break
io.json_io.jsonSave(save_json_data, f"{json_path}/change.json")