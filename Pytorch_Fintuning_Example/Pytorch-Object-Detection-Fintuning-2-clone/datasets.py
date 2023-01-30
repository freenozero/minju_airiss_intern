import os

import numpy as np
import torch

from PIL import Image
from abc import ABCMeta, abstractmethod

from detection import transforms as T
import detection.utils as utils

# https://boritea.tistory.com/29
# __all__: import * 했을 때 임포트 대상에서 어떤 것들을 가져와야 하는지 정해주는 변수고 모듈과 패키지에 모두 적용된다.
# from datasets import * 했다면 아래 코드는 PennFudan, output 함수만 임포트 된다.
__all__ = [
    "PennFudan", "output"
]

class Dataset(metaclass=ABCMeta):
    def __init__(self, params=None, transforms=None):
        self.params = params
        self.transforms = transforms
        self.imgs = None
        self.masks = None

class PennFudan(Dataset):
    def __init__(self, params, transforms):
        super().__init__(params=params, transforms=transforms)
        self.root = params['root']
        self.imgs_path = params['imgs_path']
        self.masks_path = params['masks_path']

        self.imgs = self._file_to_list(
            root = params['root'],
            path = params['imgs_path']
        )
        self.masks = self._file_to_list(
            root = params['root'],
            path = params['masks_path']
        )

    def __len__(self):
        return len(self.imgs)
    
    # 클래스의 인스턴스 자체도 슬라이싱할 수 있도록 만들 수 있다.
    # 리스트에서 슬라이싱을 하면 내부적으로 __getitem__ 메소드가 실행된다.
    def __getitem__(self, idx):
        img_path = self._path_to_file(
            root = self.root,
            path = self.imgs_path,
            file = self.imgs[idx]
        )

        mask_path = self._path_to_file(
            root = self.root,
            path = self.masks_path,
            file = self.masks[idx]
        )

        img = Image.open(img_path).convert("RGB")

        # 분할 마스크는 RGB로 변환하지 않는다. 왜냐하면 각 색상은 다른 인스턴스에 해당하며, 0은 배경에 해당하기 때문이다.
        mask = Image.open(mask_path)
        mask = np.array(mask)

        # np.unique: 문자열이 있다면 다 문자열로 변환한 뒤 같은 단어가 없게 만들어준다.(인스턴스들은 다른 색들로 인코딩 되어 있다.)
        obj_ids = np.unique(mask)
        # 첫번째 id는 배경이라 제거된다.
        obj_ids = obj_ids[1:]

        # 컬러 인코딩된 마스크를 바이너리 마스크 세트로 나눈다.
        masks = mask == obj_ids[:, None, None]

        # 각 마스크의 바운딩 박스 좌표를 얻는다.
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])
        
        # 모든 것을 torch.Tensor 타입으로 변환한다.
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # 객체 종류는 사람만 존재한다.
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # 모든 인스턴스는 군중(crowd) 상태가 아님을 가정한다.
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    # 정적 메소드: 클래스에서 바로 해당 메소드를 호출할 수 있고, self를 받지 않기 때문에 인스턴스 속성에는 접근할 수 없다.
    # 즉, self를 사용해서 클래스의 속성이나 함수를 사용하지 않는 의존성이 없는 경우 정의한다.
    @staticmethod
    def _file_to_list(root, path):
        out = list(sorted(os.listdir(os.path.join(root, path))))
        return out

    @staticmethod
    def _path_to_file(root, path, file):
        out = os.path.join(root, path, file)
        return out

# 데이터 증강/ 변환을 위한 함수 작성
def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def output(params):
    dataset = PennFudan(
        params,
        get_transform(train=True)
    )

    dataset_test = PennFudan(
        params,
        get_transform(train=False)
    )

    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(
        dataset,
        indices[:-50]
    )

    dataset_test = torch.utils.data.Subset(
        dataset_test,
        indices[-50:]
    )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=4,
        collate_fn=utils.collate_fn
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        collate_fn=utils.collate_fn
    )

    return data_loader, data_loader_test, dataset_test
