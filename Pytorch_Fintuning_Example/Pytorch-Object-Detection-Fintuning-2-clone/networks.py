from abc import ABCMeta, abstractmethod
import torch
import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

__all__ = [
    "Network"
]

# _~: private
class _Optimizer(metaclass=ABCMeta):
    def __init__(self, params):
        self.optimizer = None
        self.scheduler = None
    
    # @abstractmethod: 추상메서드를 선언해서 구현
    # 파생 클래스에서 해당 메서드를 구현하지 않을 때 에러를 발생시키게 된다.
    @abstractmethod
    def _args(self):
        pass

    @abstractmethod
    def _optimizer(self):
        pass

    @abstractmethod
    def _scheduler(self):
        pass

    @abstractmethod
    def build(self):
        pass

class Optimizer(_Optimizer):
    def __init__(self, params):
        super().__init__(params)
        self._args(params)

    def _args(self, params):
        self.optimizer = params['optimizer']
        self.scheduler = params['lr_scheduler']

    def _optimizer(self, name, model_parmas, lr, momentum, weight_decay):
        optimizer = torch.optim.SGD(
            params=model_parmas,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay
        )
        return optimizer

    def _scheduler(self, name, optimizer, step_size, gamma):
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer,
            step_size=step_size,
            gamma=gamma
        )
        return scheduler

    def build(self, model_parmas):
        optim = self._optimizer(
            name=self.optimizer['name'],
            model_parmas=model_parmas,
            lr=self.optimizer['lr'],
            momentum=self.optimizer['momentum'],
            weight_decay=self.optimizer['weight_decay'],
        )

        sche = self._scheduler(
            name=self.scheduler['name'],
            optimizer=optim,
            step_size=self.scheduler['step_size'],
            gamma=self.scheduler['gamma']
        )

        return optim, sche

class _Model(metaclass=ABCMeta):
    def __init__(self, params):
        self.model = None
        self.params = params

        self.net = None
        self.name = None
        self.num_classes = None
        self.hidden_layer = 256
        self.pretrained = False

    @abstractmethod
    def _args(self):
        pass

    @abstractmethod
    def _model(self):
        pass

    @abstractmethod
    def _params(self):
        pass

    @abstractmethod
    def build(self):
        pass

class FasterModel(_Model):
    def __init__(self, params):
        super().__init__(params=params)
        self._args(params)
    
    def _args(self, params):
        self.params = params

    def _model(self, net, pretrained, num_classes):
        model = torchvision.models.detection.fasterrcnn_resnet_fpn(
            net=net,
            pretrained=pretrained
        )

        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(
            in_channels=in_features,
            num_classes=num_classes
        )

        return model

    def _params(self, model):
        params = [p for p in model.parameters() if p.requires_grad]
        return params
    
    def build(self):
        model = self._model(
            net=self.params['net'],
            pretrained=self.params['pretrained'],
            num_classes=self.params['num_classes']
        )
        params = self._params(
            model=model
        )

        return model, params

class MaskModel(_Model):
    def __init__(self, params):
        super().__init__(params=params)
        self._args(params)
    
    def _args(self, params):
        self.params = params
    
    def _model(self, net, pretrained, hidden_layer, num_classes):
        model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(
            net=net,
            pretrained=pretrained
        )

        in_features = model.roi_heads.box_predictor.cls_score.in_features

        model.roi_heads.box_predictor = FastRCNNPredictor(
            in_channels=in_features,
            num_classes=num_classes
        )

        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_channels=in_features_mask,
            dim_reduced=hidden_layer,
            num_classes=num_classes
        )
        return model

    def _params(self, model):
        params = [p for p in model.parameters() if p.requires_grad]
        return params

    def build(self):
        model = self._model(
            net=self.params['net'],
            pretrained=self.params['pretrained'],
            hidden_layer=self.params['hidden_layer'],
            num_classes=self.params['num_classes']
        )
        params = self._params(
            model=model
        )

        return model, params

class Network:
    def __init__(self, params):
        self._args(params=params)
    
    def _args(self, params):
        name = params['Model']['name']

        if name in 'faster':
            model, model_params = self._faster(params['Model'])
        if name in 'mask':
            model, model_params = self._masks(params['Model'])
        
        model.to(self._device())

        optim = Optimizer(params=params['Optimizer'])
        optimizer, scheduler = optim.build(
            model_parmas=model_params
        )
        # property화
        self.model = model
        self.model_params = model_params
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = self._device()
    
    def _faster(self, params):
        faster_model = FasterModel(
            params=params
        )
        model, model_params = faster_model.build()
        return model, model_params

    def _masks(self, params):
        mask_model = MaskModel(
            params=params
        )

        model, model_params = mask_model.build()
        return model, model_params

    def _device(self):
        device = torch.device('cpu')
        if torch.cuda.is_available():
            device = torch.device('cuda')
        return device