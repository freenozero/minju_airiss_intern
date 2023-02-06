absolute_path = "D:/wp/data/manipulation_image/high"

name = f"high"

dirs = {
    "load_predict_image_path": f"{absolute_path}/test/jitter_image",
    "save_predict_image_path": f"{absolute_path}/test/result"
}

files = {
    "load_model_path" : f"{absolute_path}/model/save/lastest.pth",
    "load_categories_path" : f"{absolute_path}/model/save/categories.json"
}

model = {
    "min_score" : 0.5
}

