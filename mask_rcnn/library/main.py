from library.utils.header import *
from library.utils.io import *

from library.utils.log import logger
from library.utils.view import *

from library.config import train_configs as tr_cfg
from library.config import predict_configs as pr_cfg


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
            evaluate(mdl, val_dl, device=self.device)

            save_torch = {
                "epoch":epoch,
                "categories":categories["categories"],
                "weight":mdl.state_dict(),
                "optimizer":optm.state_dict(),
                "lr_scheduler":lr_sch.state_dict()
            }
            
            torch.save(save_torch, f"{self.save_model_path}/epoch_{epoch}.pth" )
            torch.save(save_torch, f"{self.save_model_path}/lastest.pth" )
        
        logger.info("Train End")

class Predict:
    def __init__(self):
        self.device = self._device()
        
        self.load_predict_image_path = pr_cfg["dirs"]["load_predict_image_path"]
        self.load_predict_image_file_list = os.listdir(pr_cfg["dirs"]["load_predict_image_path"])
        self.save_predict_image_path = pr_cfg["dirs"]["save_predict_image_path"]
        
        self.categories = self._load_categories(pr_cfg["files"]["load_categories_path"])
        self.actual_json = self._load_actual(pr_cfg["files"]["load_actural_json_path"])
        
        self.model = self._load_model(pr_cfg["files"]["load_model_path"])
        
        self.min_score = pr_cfg["model"]["min_score"]
        self.view = View()
    
    def _init_json_data(self):
        return {
            "boxes":[],
            "masks":[],
            "scores":[],
            "labels":[]
        }
    
    def _device(self):
        return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
    def _load_categories(self, full):
        path, name = split_path_name(full)
        return load_json(path, name)["categories"]
    
    def _load_actual(self, full):
        path, name = split_path_name(full)
        return load_json(path, name)

    def _load_model(self, full):    
        checkpoint = torch.load(full)
        self.categories = checkpoint["categories"]
        model = Model()
        num_classes = 1 + len(self.categories)
        
        mdl = model.build(num_classes)
        mdl.load_state_dict(checkpoint["weight"])
        mdl.to(self.device)
        mdl.eval()
        return mdl
    
    def _gpu_to_cpu_numpy(self, obj, key, idx):
        #return obj[key][idx].cpu().numpy()
        return np.round(obj[key][idx].cpu().numpy()).astype(np.uint8) if key == "masks" else obj[key][idx].cpu().numpy()
            
    def _predict_to_dataset(self, file_name, image_data, json_data):
        image_id = 0
        height, width = image_data.shape[:2]
        
        images = self._convert_predict_to_images(
            image_id=image_id, 
            path=self.save_predict_image_path, 
            file_name=file_name, 
            width=width, 
            height=height
        )
        
        annotations = self._convert_predict_to_annotations(
            image_id, 
            json_data
        )
        
        coco_json = {
            "images":images,
            "categories":self.categories,
            "annotations":annotations
        }
        
        return coco_json
        
    def _convert_predict_to_images(self, image_id, path, file_name, width, height):
        return [{
            "id": image_id,
            "dataset_id":0,
            "path":f"{path}/{file_name}",
            "file_name": f"{file_name}",
            "width":width,
            "height":height
        }]

    def _convert_box_to_bbox(self, box):
        box = np.around(box, 1)
        return [ int(box[0]), int(box[1]), int(box[2]-box[0]), int(box[3]-box[1])]
    
    def _convert_box_to_mask(self, box):
        box = np.around(box, 1)
        return [ int(box[0]), int(box[1]), int(box[2]+box[0]), int(box[3]+box[1])]

    def _convert_mask_to_segmentation(self, mask):
        mask = np.round(mask)
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_L1)
        #contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        #contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        segmentation = [ contour.flatten().tolist() for contour in contours if contour.size >= 6]
        area = int(np.sum(mask))
        
        return segmentation, area
    
    def _convert_mask_to_encode(self, mask):
        encode = maskUtils.encode(np.asfortranarray(mask))
        return encode
    
    def _convert_predict_to_annotations(self, image_id, json_data):
        masks = json_data["masks"]
        scores = json_data["scores"]
        boxes = json_data["boxes"]
        labels = json_data["labels"]
        
        annotations = []
        for idx, (mask, score, box, label) in enumerate(zip(masks, scores, boxes, labels)):
            bbox = self._convert_box_to_bbox(box)
            segmentation, area = self._convert_mask_to_segmentation(mask[0,:,:])
            
            annotations.append({
                "id":idx,
                "image_id":image_id,
                "category_id":int(label),
                "bbox":bbox,
                "score":score.tolist(),
                "segmentation":segmentation,
                "area":area,
                "iscrowd":False,
            })

        return annotations

    def _make_true_list(self, predict_image_file):
        '''return true_list ([[category_id:bbox], ...])'''
        true_list = []
        for ann in self.actual_json["annotations"]:
            if ann["image_id"] == predict_image_file:
                true_list.append({ann["category_id"]: ann["bbox"]})
        return true_list
    
    def _confusion_matrix(self, pred_list, true_list):

        result = [[0, 0, 0, 0] for x in range(4)] # [TP:0, FP:1, FN:2, TN:3]
        # True Positive(TP) : 실제 True인 정답을 True라고 예측 (정답)
        # False Positive(FP) : 실제 False인 정답을 True라고 예측 (오답)
        # False Negative(FN) : 실제 True인 정답을 False라고 예측 (오답)
        # True Negative(TN) : 실제 False인 정답을 False라고 예측 (정답)

        true_list_copy = copy.deepcopy(true_list)
        pred_list_copy = copy.deepcopy(pred_list)
        for pred in pred_list:
            pred_boxes = list(pred.values())[0]
            pred_keys = list(pred.keys())[0]
            for true in true_list:
                true_boxes = self._convert_box_to_mask(list(true.values())[0])
                true_keys = list(true.keys())[0]
                iou = self._iou(pred_boxes, true_boxes)
                # iou도 맞고 카테고리도 맞는 것
                if(iou >= self.min_score and pred_keys == true_keys):
                   
                    if true in true_list_copy and pred in pred_list_copy:
                        del true_list_copy[true_list_copy.index(true)]
                        del pred_list_copy[pred_list_copy.index(pred)]
                        result[pred_keys-1][0]+=1 # TP
                        break
                # iou는 맞지만 카테고리가 다른 것
                elif(iou >= self.min_score and pred_keys != true_keys):
                    
                    if true in true_list_copy and pred in pred_list_copy:
                        del true_list_copy[true_list_copy.index(true)]
                        del pred_list_copy[pred_list_copy.index(pred)]
                        result[pred_keys-1][1]+=1 # FP
                        break

        # 예측했지만 정답이 아닌 것들
        for pred in pred_list_copy:
            pred_keys = list(pred.keys())[0] # FP
            result[pred_keys-1][1] += 1
        
        # 정답이지만 예측되지 않은 것들
        for true in true_list_copy:
            true_keys = list(true.keys())[0] # FN
            result[true_keys-1][2] += 1
        # print(result)
        return result
    
    def _iou(self, pred_boxes, true_boxes):
        box1_area = (pred_boxes[2] - pred_boxes[0] + 1) * (pred_boxes[3] - pred_boxes[1] + 1)
        box2_area = (true_boxes[2] - true_boxes[0] + 1) * (true_boxes[3] - true_boxes[1] + 1)

        x1 = max(pred_boxes[0], true_boxes[0])
        y1 = max(pred_boxes[1], true_boxes[1])
        x2 = min(pred_boxes[2], true_boxes[2])
        y2 = min(pred_boxes[3], true_boxes[3])

        w = max(0, x2 - x1 + 1)
        h = max(0, y2 - y1 + 1)

        inter = w * h

        # x1, y1, x2, y2, w, h, inter = int(x1), int(y1), int(x2), int(y2), int(w), int(h), int(inter)

        iou = inter / (box1_area + box2_area - inter)
        return iou
    
    def _convert_precision(self, l):
        '''[TP:0, FP:1, FN:2, TN:3]'''
        return l[0]/(l[0]+l[1])

    def _convert_recall(self, l):
        '''[TP:0, FP:1, FN:2, TN:3]'''
        return l[0]/(l[0]+l[2])

    def start(self):
        to_tensor = ToTensor()
        result = [[0, 0, 0, 0] for x in range(4)] # [TP:0, FN:1, FP:2, TN:3]

        for _, predict_image_file in enumerate(self.load_predict_image_file_list):             
            # this list is limited predict_image_file
            pred_list = [] # [[category_id:bbox], ...]

            true_list = self._make_true_list(int(predict_image_file.rstrip('.png')))

            #
            load_path = self.load_predict_image_path
            load_file_name = predict_image_file
            
            save_path = self.save_predict_image_path
            save_json_file_name = f"{split_extension_file_name(predict_image_file)}.json"
            #

            image_data = cv_load_image(load_path, load_file_name, cv2.IMREAD_COLOR)
            image_tensor = to_tensor(image_data)
            
            with torch.no_grad():
                predictions = self.model([image_tensor.to(self.device)])[0]
            
            json_data = self._init_json_data()
            for idx, score in enumerate(predictions["scores"]):
                if float(score) > self.min_score:
                    pred_list.append({int(predictions["labels"][idx]): predictions["boxes"][idx].tolist()})
                    for key in predictions.keys():
                        json_data[key].append(self._gpu_to_cpu_numpy(predictions, key, idx))

            coco_data = self._predict_to_dataset(
                file_name=predict_image_file, 
                image_data=image_data,
                json_data=json_data
            )

            save_json(coco_data, save_path, save_json_file_name)
            result2 = self._confusion_matrix(pred_list, true_list)
            result = np.array(result) + np.array(result2)
            # print(f"knife precision: {self._convert_precision(result[0])}")
            # print(f"gun precision: {self._convert_precision(result[1])}")
            # print(f"battery precision: {self._convert_precision(result[2])}")
            # print(f"laserpointer precision: {self._convert_precision(result[3])}")

                    
            # print(f"knife recall: {self._convert_recall(result[0])}")
            # print(f"gun recall: {self._convert_recall(result[1])}")
            # print(f"battery recall: {self._convert_recall(result[2])}")
            # print(f"laserpointer recall: {self._convert_recall(result[3])}")
                
            self.view.visualize(save_path, load_file_name, image_data, coco_data)
        
        print(f"knife precision: {self._convert_precision(result[0])}")
        print(f"gun precision: {self._convert_precision(result[1])}")
        print(f"battery precision: {self._convert_precision(result[2])}")
        print(f"laserpointer precision: {self._convert_precision(result[3])}")

                
        print(f"knife recall: {self._convert_recall(result[0])}")
        print(f"gun recall: {self._convert_recall(result[1])}")
        print(f"battery recall: {self._convert_recall(result[2])}")
        print(f"laserpointer recall: {self._convert_recall(result[3])}")