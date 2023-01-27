import os

import numpy as np
import torch

from PIL import Image
from abc import ABCMeta, abstractmethod

from detection import transforms as T
import detection.utils as utils

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
            root=params['root'],
            path=params['imgs_path']
        )
        self.masks = self._file_to_list(
            root=params['root'],
            path=params['masks_path']
        )

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = self._path_to_file(
            root=self.root,
            path=self.imgs_path,
            file=self.imgs[idx]
        )

        mask_path = self._path_to_file(
            root=self.root,
            path=self.masks_path,
            file=self.masks[idx]
        )

        img = Image.open(img_path).convert("RGB")

        mask = Image.open(mask_path)
        mask = np.array(mask)

        obj_ids = np.unique(mask)
        obj_ids = obj_ids[1:]

        masks = mask == obj_ids[:, None, None]

        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

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

    @staticmethod
    def _file_to_list(root, path):
        out = list(sorted(os.listdir(os.path.join(root, path))))
        return out

    @staticmethod
    def _path_to_file(root, path, file):
        out = os.path.join(root, path, file)
        return out


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
