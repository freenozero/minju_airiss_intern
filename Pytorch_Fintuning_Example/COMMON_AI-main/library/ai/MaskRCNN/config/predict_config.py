import datetime

name = f"cow"

absolute_path = f"D:/workspace/COMMON_AI/sample/ai/example_1"

dirs = {
    "load_predict_image_path": f"{absolute_path}/predict/image",
    "save_predict_image_path": f"{absolute_path}/predict/result"
}

files = {
    "load_model_path" : f"{absolute_path}/model/load/{name}/lastest.pth",
    "load_categories_path" : f"{absolute_path}/model/load/{name}/categories.json"
}

model = {
    "min_score" : 0.5
}

