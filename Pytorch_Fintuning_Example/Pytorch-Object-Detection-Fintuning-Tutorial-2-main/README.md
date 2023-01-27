# Faster_Mask_RCNN

### pytorch obeject detection finetuning tutorial

#### python : 3.7.5

#### torch : 1.8.1+cu111

#### pycocotools : 2.0.2
  
  - faster & mask R-CNN 튜닝

1. Struct
  ```
  Faster_Mask_RCNN
  | detection
  └─| coco_eval.py
    | coco_utils.py
    │ engine.py
    │ group_by_aspect_ratio.py
    │ presets.py
    │ train.py
    │ transforms.py
    │ utils.py
  | PenFudanPed
  | active.py
  | datasets.py
  | networks.py
  | run.py
  ```
  - detection :  pytorch에서 기본으로 재공해주는 라이브러리 예시
  - PenFudanPed : 학습 및 테스트를 위한 영상 데이터 폴더 : [ PenFudanPed - PASCAL Annotation Version 1.00 ](https://www.cis.upenn.edu/~jshi/ped_html/)
  - active.py : Trian/Predict/View
  - datasets.py : 데이터 전처리
  - networks.py : 모델 생성
  - run.py : 실행


2. Setting
  - 가상환경(venv) 파일 수정
    - venv\lib\site-packages\torchvision\models\detection\faster_rcnn.py
    - venv\lib\site-packages\torchvision\models\detection\mask_rcnn.py

  - import 수정 

  ###### faster_rcnn.py 수정
  ~~~python
  ...
  # 17 line
  __all__ = [
      "FasterRCNN", "fasterrcnn_resnet_fpn", "fasterrcnn_resnet50_fpn", "fasterrcnn_mobilenet_v3_large_320_fpn",
      "fasterrcnn_mobilenet_v3_large_fpn"
  ]

  ...
  # write
  def fasterrcnn_resnet_fpn(net='resnet50', pretrained=False, progress=True,
                            num_classes=91, pretrained_backbone=True, trainable_backbone_layers=None, **kwargs):

      trainable_backbone_layers = _validate_trainable_layers(
          pretrained or pretrained_backbone, trainable_backbone_layers, 5, 3)

      backbone = resnet_fpn_backbone(
          net, pretrained_backbone, trainable_layers=trainable_backbone_layers)
      model = FasterRCNN(backbone, num_classes, **kwargs)

      return model
  ~~~

  ###### mask_rcnn.py 수정
  ~~~python

  ...
  # 13 line
  __all__ = [
      "MaskRCNN", "maskrcnn_resnet_fpn", "maskrcnn_resnet50_fpn",
  ]

  ...


  def maskrcnn_resnet_fpn(net='resnet50', pretrained=False, progress=True,
                          num_classes=91, pretrained_backbone=True, trainable_backbone_layers=None, **kwargs):
      trainable_backbone_layers = _validate_trainable_layers(
          pretrained or pretrained_backbone, trainable_backbone_layers, 5, 3)

      backbone = resnet_fpn_backbone(
          net, pretrained_backbone, trainable_layers=trainable_backbone_layers)
      model = MaskRCNN(backbone, num_classes, **kwargs)

      return model

  ~~~

  ###### detection의 import수정( coco_eval.py, coco_utils.py, engine.py, presets.py, train.py )
  ```
    import utils => from . import utils
    import transforms as T => from . import transforms as T
    import presets = > from . import presets
    from coco_utils => from .coco_utils
    from coco_eval => from .coco_eval
    from group_by_aspect_ratio => from .group_by_aspect_ratio
    from engine => from .engine
  ```


3. [git repository](https://github.com/MizzleAa/Faster_Mask_RCNN)

4. Run : python run.py

5. Result  
- image  
![result_15](/result/15.png)  
  
- masks  
![result_15_0](/result/15_0.png)
![result_15_0](/result/15_1.png)
![result_15_0](/result/15_2.png)
![result_15_0](/result/15_3.png)
![result_15_0](/result/15_4.png)
