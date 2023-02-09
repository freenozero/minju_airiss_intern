name = f"highlow"

model_path = f"D:/wp/data/manipulation_image(9,1)/{name}"
test_path = "D:/wp/data/test_image/High_energy_image_size(700x700)_16bit"

dirs = {
    "load_predict_image_path": f"{test_path}/image",
    "save_predict_image_path": f"{test_path}/result"
}

files = {
    "load_actural_json_path" : f"{test_path}/json/change.json",
    "load_model_path" : f"{model_path}/model/save/lastest.pth",
    "load_categories_path" : f"{model_path}/model/save/categories.json"
}

model = {
    "min_score" : 0.999
}