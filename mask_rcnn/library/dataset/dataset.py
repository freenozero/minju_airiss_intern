from library.utils.header import *
from library.utils.io import *

class COCODataset(torch.utils.data.Dataset):
    
    def __init__(self, absolute_path, coco, transforms):
        self.coco = coco
        self.transforms = transforms
        self.absolute_path = absolute_path
        self.image_count = len(coco.getImgIds())
        
    def __getitem__(self, idx):
        
        image_ids = self.coco.getImgIds()[idx]
        image = self.coco.loadImgs(image_ids)[0]
        image_data = Image.open(f"{self.absolute_path}/{image['file_name']}")
        
        annotation_ids = self.coco.getAnnIds(imgIds=image_ids)
        annotations = self.coco.loadAnns(annotation_ids)
        
        target = {
            "image_id":[image_ids],
            "labels":[],
            "boxes":[],
            "masks":[],
            "area":[],
            "iscrowd":[]
        }
        
        for ann in annotations:
            target["labels"].append(ann["category_id"])
            target["boxes"].append(self._bbox_to_point(ann["bbox"]))

            # seg가 [[seg]]일 경우
            # ann2 = copy.deepcopy(ann)
            # ann2["segmentation"] = ann2["segmentation"][0]
            
            # seg가 [seg]일 경우
            target["masks"].append(self.coco.annToMask(ann))

            target["area"].append(ann["area"])
            target["iscrowd"].append(ann["iscrowd"])
        
        target["image_id"] = torch.as_tensor(target["image_id"])
        target["labels"] = torch.as_tensor(target["labels"])
        target["boxes"] = torch.as_tensor(target["boxes"])
        target["masks"] = torch.as_tensor(np.array(target["masks"]))
        target["area"] = torch.as_tensor(target["area"])
        target["iscrowd"] = torch.as_tensor(target["iscrowd"])
        
        if self.transforms is not None:
            image_data, target = self.transforms(image_data, target)
        
        return image_data, target

    def __len__(self):
        return self.image_count
    
        
    def _bbox_to_point(self, bbox):
        return [bbox[0],bbox[1],bbox[0]+bbox[2],bbox[1]+bbox[3]]