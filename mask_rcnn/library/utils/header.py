import cv2
import os
import json
import copy

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from PIL import ImageFont, ImageDraw, Image

import torch
import torchvision


from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.transforms import ToTensor, ToPILImage

from torchvision import transforms as T

from torch.optim.lr_scheduler import StepLR

from library.vision.references.detection.engine import train_one_epoch, evaluate
import library.vision.references.detection.utils as utils
import library.vision.references.detection.transforms as T

from pycocotools.coco import COCO
from pycocotools import mask as maskUtils