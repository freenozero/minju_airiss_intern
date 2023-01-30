from datasets import output as dataset_output
from networks import Network
from active import Train, Predict, View

def main():
    # 데이터 정의
    params = {
        'root': 'D:\wp\minju_airiss_intern\Pytorch_Fintuning_Example\Pytorch-Object-Detection-Fintuning-Tutorial-2-main\PennFudanPed',
        'imgs_path': 'PNGImages',
        'masks_path': 'PedMasks'
    }
    data_loader, data_loader_test, dataset_test = dataset_output(params)

    params = {
        'Model' : {
            'net' : 'mask_rcnn',
            'name' : 'mask',
            'num_classes' : 2,
            'hidden_layer' : 256,
            'pretrained' : True
        },
        'Optimizer' : {
            'optimizer': {
                'name' : 'sgd',
                'lr' : 0.001,
                'momentum' : 0.9,
                'weight_decay' : 0.0005
            },
            'lr_scheduler' : {
                'name' : 'steplr',
                'step_size' : 3,
                'gamma' : 0.1
            }
        }
    }
    net = Network(params)

    # 학습 옵션 정의
    params = {
        'num_epochs': 10,
        'model': net.model,
        'params': net.model_params,
        'optimizer': net.optimizer,
        'scheduler': net.scheduler,
        'device': net.device,
        'data_loader': data_loader,
        'data_loader_test': data_loader_test
    }
    # 학습
    train = Train(params)
    train.run()

    # 예측
    params = {
        'model' : net.model,
        'dataset_test' : dataset_test,
        'device' : net.device,
        'min_scores' : 0.8
    }
    predict = Predict(params)
    images, boxes, masks, scores, labels = predict.run()

    # 데이터 저장
    params = {
        'images': images,
        'boxes': boxes,
        'masks': masks,
        'scores': scores,
        'labels': labels
    }
    view = View(params)
    view.view()

if __name__ == "__main__":
    main()