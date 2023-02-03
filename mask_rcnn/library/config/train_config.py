absolute_path = "D:/wp/data/manipulation_image/high"

name = f"high"
dirs = {
    'train_image_path': f"{absolute_path}/train/jitter_image",
    'val_image_path': f"{absolute_path}/val/jitter_image",
    'save_model_path': f"{absolute_path}/model/save",
    'load_model_path': f"{absolute_path}/model/load"
}

files = {
    'load_model_path': f"{absolute_path}/model/load/lastest.pth",
    'train_json_path': f"{absolute_path}/train/json/data.json",
    'val_json_path': f"{absolute_path}/val/json/data.json",
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
    "epochs": 1,
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