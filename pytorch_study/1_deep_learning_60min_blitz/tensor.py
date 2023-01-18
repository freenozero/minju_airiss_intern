import torch
import numpy as np

# # 텐서 초기화
# ## 데이터로부터 직접 생성
# data = [[1, 2], [3, 4]]
# x_data = torch.tensor(data)
# print(x_data)

# ## NumPy 배열로부터 생성
# np_array = np.array(data)
# x_np = torch.from_numpy(np_array)
# print(x_np)

# ## 다른 텐서로부터 생성
# x_ones = torch.ones_like(x_data)
# print(f"Ones Tensor: \n {x_ones}\n")
# x_rand = torch.rand_like(x_data, dtype=torch.float)
# print(f"Random Tensor: \n {x_rand}\n")

# ## 무작위(random) 또는 상수(constant) 값 사용
# shape = (2, 3)
# rand_tensor = torch.rand(shape)
# ones_tensor = torch.ones(shape)
# zeros_tensor = torch.zeros(shape)
# print(f"Random Tensor: \n {rand_tensor}\n")
# print(f"Ones Tensor: \n {ones_tensor}\n")
# print(f"Zeros Tensor: \n {zeros_tensor}\n")





# # tensor attribute
# tensor = torch.rand(3, 4)
# print(f"Shape of tensor: {tensor.shape}")
# print(f"Datatype of tensor: {tensor.dtype}")
# print(f"Device tensor is stored on: {tensor.device}")

# # tensor operation
# ## GPU 할당
# if torch.cuda.is_available():
#   tensor = tensor.to('cuda')
#   print(f"Device tensor is stored on: {tensor.device}")

# ## Numpy식의 표준 인덱싱과 슬라이싱
# tensor = torch.ones(4, 4)
# tensor[:,1] = 0
# print(tensor)

# ## 텐서 합치기
# t1 = torch.cat([tensor, tensor, tensor], dim=1)
# print(t1)

# ## 텐서 곱하기
# print(f"tensor.mul(tensor) \n {tensor.mul(tensor)} \n")
# #### 다른 문법
# print(f"tensor * tensor \n {tensor * tensor}")

# ### 행렬곱
# print(f"tensor.matmul(tensor.T) \n {tensor.matmul(tensor.T)} \n")
# #### 다른 문법
# print(f"tensor @ tensor.T \n {tensor @ tensor.T}")

# ## 바꿔치기(in-place) 연산
# print(tensor, "\n")
# tensor.add_(5)
# print(tensor)





# Numpy 변환(Bridge)
## 텐서를 Numpy 배열로 변환
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")
### 텐서 변경시 numpy 배열에 반영
t.add_(1)
print(f"t: {t}")
print(f"n: {n}")

## Numpy 배열을 텐서로 변환
n = np.ones(5)
t = torch.from_numpy(n)
### Numpy 변경시 텐서에 반영
np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")

