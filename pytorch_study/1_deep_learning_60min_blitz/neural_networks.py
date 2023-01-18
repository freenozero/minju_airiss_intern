# https://tutorials.pytorch.kr/beginner/blitz/neural_networks_tutorial.html#sphx-glr-beginner-blitz-neural-networks-tutorial-py
# 예시: 숫자 이미지 분류

import torch
# torch.nn은 미니배치만 지원
import torch.nn as nn
import torch.nn.functional as F

# 신경망 정의
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 입력 이미지 채널 1개, 출력 채널 6개, 5x5의 정사각 컨볼루션 행렬
        # 컨볼루션 커널 정의
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)

        # 아핀(affine) 연산: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # (2, 2) 크기 윈도우에 대해 max pooling
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # 크기가 제곱수라면, 하나의 숫자만을 특정(specify)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
net = Net()
print(net)

# 모델의 학습 가능한 매개변수는 net.parameters()에 의해 반환
params = list(net.parameters())
print(len(params))
print(params[0].size())

# 임의의 32x32 입력값 넣기
# Note: LeNet의 예상 입력 크기는 32x32, 이 신경망에 MNIST 데이터 셋을 사용하기 위해서는 데이터 셋의 이미지 크기를 32x32로 변경해야 함
input = torch.randn(1, 1, 32, 32)
out = net(input)
print(out)

# 모든 매개변수의 변화도 버퍼(gradient buffer)를 0으로 설정하고 무작위 값으로 역전파 진행
net.zero_grad()
out.backward(torch.randn(1, 10))



# Loss Function
## Example: MSE
output = net(input)
target = torch.randn(10) # 임의의 정답
target = target.view(1, -1) # 출력과 같은 shape로 만듦
criterion = nn.MSELoss()

loss = criterion(output, target)
print(f"loss: {loss}")



# Backprop
## eror 역전파: loss.backward()
## 기존에 계산된 변화도의 값을 누적 시키고 싶지 않다면 기존에 계산된 변화도를 0으로 만드는 작업이 필요
## loss.backward()를 호출하여 역전파 전과 후에 conv1의 bias 변수의 변화도를 살펴보기
net.zero_grad() # 모든 매개변수의 변화도 버퍼를 0으로 만듦
print('conv.bias.grad befor backward')
print(net.conv1.bias.grad)

loss.backward()

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)



# 가중치 갱신
## Example: SGD(확률적 경사하강법):: 새로운 가중치(weight) = 가중치(weight) - 학습률(learning rate) * 변화도(gradient)
learning_rate = 0.01
for f in net.parameters():
    f.data.sub_(f.grad.data * learning_rate)



## torch.optim: 신경망 구성시 SGD, Nesterov-SGD, Adam, RMSProp 등의 갱신 규칙 사용
import torch.optim as optim

# Optimzer를 생성
optimizer = optim.SGD(net.parameters(), lr=0.01)

# 학습 과정(traning loop)
optimizer.zero_grad() # 변화율 버퍼를 0으로
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step() # 업데이트 진행
