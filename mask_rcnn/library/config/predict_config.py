name = f"highlow"

model_path = f"D:/wp/data/manipulation_image(9,1)/{name}"
test_path = "D:/wp/data/test_image/High_energy_image_size(700x700)_16bit"
# test_path = f"D:/wp/data/manipulation_image(8,1,1)/{name}/test"

dirs = {
    "load_predict_image_path": f"{test_path}/image",
    # "load_predict_image_path": f"{test_path}/jitter_image",
    "save_predict_image_path": f"{test_path}/result"
}

files = {
    # "load_actural_json_path" : f"{test_path}/json/data.json",
    "load_actural_json_path" : f"{test_path}/json/change.json",
    "load_model_path" : f"{model_path}/model/save/lastest.pth",
    "load_categories_path" : f"{model_path}/model/save/categories.json"
}

model = {
    "min_score" : 0.5
}