import datetime

name = f"cow"

absolute_path = f"D:/workspace/COMMON_AI/sample/ai/example_1"

dirs = {
    "train_image_path": f"{absolute_path}/val/image",
    "val_image_path": f"{absolute_path}/val/image",
    "save_model_path" : f"{absolute_path}/model/save/{name}",
    "load_model_path" : f"{absolute_path}/model/load/{name}"
}

files = {
    "load_model_path" : f"{absolute_path}/model/load/{name}/lastest.pth",
    "train_json_path": f"{absolute_path}/val/json/data.json",
    "val_json_path": f"{absolute_path}/val/json/data.json",
}

dataloader = {
    "train" : {
        "batch_size":4,
        "shuffle":True,
        "num_workers":0,
    },
    
    "val": {
        "batch_size":1,
        "shuffle":False,
        "num_workers":0,
    }
}

model = {
    "epochs": 40,
    "detection":{
        "hidden_layer": 256,
        "pretrained": True
    },
    
    "optimizer" : {
        "lr":0.005,
        "momentum":0.9
    },
    "lr_scheduler" : {
        "step_size":3,
        "gamma":0.1
    }
}