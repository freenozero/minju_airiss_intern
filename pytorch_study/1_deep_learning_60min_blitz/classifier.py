# https://tutorials.pytorch.kr/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py

# 이미지 분류기 학습
## 1. CIFAR10을 불러오고 정규화
### torchvision을 사용하여 쉽게 CIFAR10을 불러올 수 있음
import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# # 학습용 이미지 show
import matplotlib.pyplot as plt
import numpy as np

def imshow(img):
    img = img / 2 + 0.5 # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# # 학습용 이미지를 무작위로 가져오기
# dataiter = iter(trainloader)
# images, labels = dataiter.next()

# # 이미지 보여주기
# imshow(torchvision.utils.make_grid(images))
# # 정답(label) 출력
# print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))



## 2. 합성곱(CNN) 정의
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()

## 3. 손실 함수와 Optimizer 정의
### 교차 엔트로피 손실(Cross-Entropy loss)와 모멘텀(momentum) 값을 갖는 SGD를 사용
import torch.optim as optim
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

## 4. 신경망 학습
### 데이터를 반복해서 신경망에 입력으로 제공하고, 최적화(Optimize)
for epoch in range(2):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # [inputs, labes]의 목록인 data로부터 입력을 받음
        inputs, labels = data

        # 변화도(gradient) 매개변수를 0으로 만듦
        optimizer.zero_grad()

        # 순전파 + 역전파 + 최적화를 한 후
        output = net(inputs)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        # 통계 출력
        running_loss += loss.item()
        if i % 2000 == 1999:
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')

### 학습한 모델 저장
PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)

## 5. 시험용 데이터로 신경망 검사하기
dataiter = iter(testloader)
images, labels = dataiter.next()

# 이미지를 출력합니다.
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))