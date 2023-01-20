# https://tutorials.pytorch.kr/beginner/blitz/autograd_tutorial.html#sphx-glr-beginner-blitz-autograd-tutorial-py
# torch.autograde = 신경망 학습을 지원하는 pytorch의 자동 미분 엔진

import torch, torchvision

model = torchvision.models.resnet18(pretrained=True)
data = torch.rand(1, 3, 64, 64)
labels = torch.rand(1, 1000)

# 순전파 단계(input data를 모델의 각 layer에 통과시켜 prediction을 생성)
prediction = model(data)

# 모델의 예측값과 그에 해당하는 label(정답)을 사용하여 error를 계산
loss = (prediction - labels).sum()

# 역전파 단계(error tensor에 .backward()를 호출하면 역전파가 시작, autograd가 매개변수의 .grad 속성에 모델의 각 매개변수에 대한 변화도(gradient)를 계산하고 저장)
loss.backward()

# optimizer를 불러옴(ex는 learning rate = 0.1과 momentum = 0.9를 갖는 SGD)
optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)

# last: .step을 호출해서 gradient descent를 시작, 옵티마이저는 .grad에 저장된 변화에 따라 각 매개 변수를 조정(adjust)함
optim.step()