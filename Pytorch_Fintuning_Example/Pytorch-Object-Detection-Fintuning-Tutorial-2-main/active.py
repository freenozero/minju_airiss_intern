from detection.engine import evaluate, train_one_epoch

import torch

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

import os
import time

__all__ = [
    "Train", "Predict", "View"
]


class Train:
    def __init__(self, params):
        self.num_epochs = params['num_epochs']
        self.params = params['params']
        self.model = params['model']
        self.optimizer = params['optimizer']
        self.scheduler = params['scheduler']
        self.device = params['device']
        self.data_loader = params['data_loader']
        self.data_loader_test = params['data_loader_test']

    def run(self):
        for epoch in range(self.num_epochs):
            # print(epoch)
            train_one_epoch(self.model, self.optimizer, self.data_loader,
                            self.device, epoch, print_freq=5)
            self.scheduler.step()
            evaluate(self.model, self.data_loader_test, device=self.device)


class Predict:
    def __init__(self, params):
        self.model = params['model']
        self.dataset_test = params['dataset_test']
        self.device = params['device']
        self.min_scores = params['min_scores']

    def run(self):
        images = []
        boxes = []
        masks = []
        scores = []
        labels = []

        self.model.eval()
        for img, _ in self.dataset_test:

            with torch.no_grad():
                prediction = self.model([img.to(self.device)])[0]

            mask = []
            box = []
            score = []
            label = []

            for idx, sco in enumerate(prediction['scores']):
                sco = float(sco)
                if sco > self.min_scores:
                    if 'masks' in prediction:
                        mask.append(
                            prediction['masks'][idx].mul(
                                255).byte().cpu().numpy()
                        )
                    box.append(
                        prediction['boxes'][idx].cpu().numpy()
                    )
                    label.append(
                        prediction['labels'][idx].cpu().numpy()
                    )

                    sco = round(sco*100, 2)
                    score.append(sco)

            print(f"scores = {score}")
            image = img.mul(255).permute(1, 2, 0).byte().numpy()

            boxes.append(box)
            masks.append(mask)
            images.append(image)
            scores.append(score)
            labels.append(label)

        return images, boxes, masks, scores, labels


class View:
    def __init__(self, params):
        self.images = params['images']
        self.boxes = params['boxes']
        self.masks = params['masks']
        self.scores = params['scores']
        self.labels = params['labels']

        self.mkdir()

    def mkdir(self):
        self.path = f"./save_{time.time()}"
        os.mkdir(self.path)

    def view(self):
        css_font = {
            'family': 'serif',
            'color':  'red',
            'weight': 'normal',
            'size': 7
        }
        css_box = {
            'boxstyle': 'round',
            'ec': (1.0, 0.5, 0.5),
            'fc': (1.0, 1.0, 1.0)
        }

        for idx, image in enumerate(self.images):
            plt.cla()
            plt.imshow(image)
            ax = plt.gca()

            m_idx = 0
            for mask in self.masks[idx]:
                if mask is not None:
                    file_path_name = f"{self.path}/{idx}_{m_idx}.png"
                    Image.fromarray(mask[0]).save(file_path_name)
                    m_idx += 1

            for box, score, label in zip(self.boxes[idx], self.scores[idx], self.labels[idx]):
                plt.text(
                    box[0],
                    box[1],
                    f"{label} : {score}",
                    fontdict=css_font,
                    bbox=css_box
                )

                rect = patches.Rectangle(
                    (box[0], box[1]),
                    (box[2]-box[0]),
                    (box[3]-box[1]),
                    linewidth=1,
                    edgecolor='red',
                    fill=False
                )
                ax.add_patch(rect)

            file_path_name = f"{self.path}/{idx}.png"
            plt.savefig(file_path_name, dpi=300)
