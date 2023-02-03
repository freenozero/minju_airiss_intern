from library.utils.header import *
from library.utils.io import *

from library.utils.log import logger

from library.config import train_configs as tr_cfg

from library.dataset.dataset import COCODataset

class DataLoader:
    
    def __init__(self, purpose):
        self.purpose = purpose
        self.is_transform = True if purpose == "train" else False
        self.image_path = tr_cfg["dirs"][f"{purpose}_image_path"]
        self.json_data = COCO(tr_cfg["files"][f"{purpose}_json_path"])
        self.batch_size = tr_cfg["dataloader"][f"{purpose}"]["batch_size"]
        self.shuffle = tr_cfg["dataloader"][f"{purpose}"]["shuffle"]
        self.num_workers = tr_cfg["dataloader"][f"{purpose}"]["num_workers"]
        
        self._info()
        
    def build(self):
        return torch.utils.data.DataLoader(
            COCODataset(
                absolute_path=self.image_path,
                coco=self.json_data,
                transforms=self._get_transform(), # transforms=self._get_transform(is_transform=self.is_transform),
            ),
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,        
            collate_fn=utils.collate_fn    
        )

    def _info(self):
        #이미지 카테고리 어노테이션 개수
        count_str = '{:<18} @[ Images={:<12d} | Categoires={:<12d} | Annotations={:<12d} ]'
        logger.info(count_str.format(
            f"{self.purpose} Count", 
            len(self.json_data.imgs.keys()), 
            len(self.json_data.cats.keys()), 
            len(self.json_data.anns.keys())
        ))
        
        #카테고리 목록 리스트
        length = len(self.json_data.cats.values())
        cats = [ (cat["id"], len(catImgs), cat["name"]) for cat, catImgs in zip(list(self.json_data.cats.values()), list(self.json_data.catToImgs.values()))]
        title_str = '{:<8s} | {:<12s} | {:<12s}'
        logger.info(
            title_str.format(
                "id",
                "count",
                "name"
            )
        )
        
        #카테고리별 어노테이션 개수
        for cat in cats:
            cat_str = '{:<8d} | {:<12d} | {:<12s}'
            logger.info(
                cat_str.format(
                    cat[0], cat[1], cat[2]
                )
            )
        
    def _get_transform(self):
        transforms = []
        transforms.append(T.ToTensor())
        
        # if is_transform:
        #     transforms.append(T.RandomHorizontalFlip(0.5))
        
        return T.Compose(transforms)

class Optimizer:
    def __init__(self):
        self.lr = tr_cfg["model"]["optimizer"]["lr"]
        self.momentum = tr_cfg["model"]["optimizer"]["momentum"]
        
        self.step_size = tr_cfg["model"]["lr_scheduler"]["step_size"]
        self.gamma = tr_cfg["model"]["lr_scheduler"]["gamma"]

    def build(self, model_params):
        optimizer = torch.optim.SGD(
            params=model_params, 
            lr=self.lr,
            momentum=self.momentum
        )
        
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer,
            step_size=self.step_size,
            gamma=self.gamma
        )

        return optimizer, lr_scheduler

class Model:
    def __init__(self):
        self.pretrained = tr_cfg["model"]["detection"]["pretrained"]
        self.hidden_layer = tr_cfg["model"]["detection"]["hidden_layer"]
        self.model = None
        
    def build(self, num_classes):
        self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=self.pretrained)
        #faster
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(
            in_channels=in_features, 
            num_classes=num_classes
        )

        #mask
        in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
        self.model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_channels=in_features_mask,
            dim_reduced=self.hidden_layer,
            num_classes=num_classes
        )

        return self.model
    
    def params(self):
        return [p for p in self.model.parameters() if p.requires_grad]

class Train:
    def __init__(self):
        self.name = tr_cfg["name"]
        self.epochs = tr_cfg["model"]["epochs"]
        self.save_model_path = tr_cfg["dirs"]["save_model_path"]
        self.load_model_path = tr_cfg["files"]["load_model_path"]
        
        self.device = self._device()
    
    def _device(self):
        return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    def start(self):
        logger.info("Train Start")
        
        # 데이터 설정
        train_data_loader = DataLoader("train")
        val_data_loader = DataLoader("val")
        
        train_dl = train_data_loader.build()
        val_dl = val_data_loader.build()

        #카테고리 정보 저장
        categories = {
            "categories" : list(train_data_loader.json_data.cats.values())
        }
        save_json(categories, f"{self.save_model_path}", "categories.json")

        #카테고리 개수
        num_classes = 1+len(categories["categories"])

        #model 설정
        model = Model()
        mdl = model.build(num_classes)
        mdl.to(self.device)
        
        # optimizer 설정
        optimizer = Optimizer()
        optm, lr_sch = optimizer.build(model_params=model.params())

        for epoch in range(self.epochs):
            train_one_epoch(mdl, optm, train_dl, self.device, epoch, print_freq=10)
            lr_sch.step()
        #     evaluate(mdl, val_dl, device=self.device)

        #     save_torch = {
        #         "epoch":epoch,
        #         "categories":categories["categories"],
        #         "weight":mdl.state_dict(),
        #         "optimizer":optm.state_dict(),
        #         "lr_scheduler":lr_sch.state_dict()
        #     }
            
        #     torch.save(save_torch, f"{self.save_model_path}/epoch_{epoch}.pth" )
        #     torch.save(save_torch, f"{self.save_model_path}/lastest.pth" )


logger.info("Train End")
