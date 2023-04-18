import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import torchvision.datasets as dsets
import torchvision.transforms as transforms

# GPU 사용
# USE_CUDA = torch.cuda.is_available() 
# device = torch.device("cuda" if USE_CUDA else "cpu")
# print("{}로 학습합니다.".format(device))

# 랜덤 시드
torch.manual_seed(1)

# 데이터
x_train = dsets.MNIST(root='MNIST_data/', # 다운로드 경로 지정
                          train=True, # True를 지정하면 훈련 데이터로 다운로드
                          transform=transforms.ToTensor(), # 텐서로 변환
                          download=True)

y_train = dsets.MNIST(root='MNIST_data/', # 다운로드 경로 지정
                         train=False, # False를 지정하면 테스트 데이터로 다운로드
                         transform=transforms.ToTensor(), # 텐서로 변환
                         download=True)

# 데이터로더
dataLoader = torch.utils.data.DataLoader(dataset=x_train, batch_size=100, shuffle=True, drop_last=True)

# 모델
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.layer1 = nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer2 = nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc = torch.nn.Linear(7 * 7 * 64, 10, bias=True)

        torch.nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

model = CNN()

# 비용함수
cost = nn.CrossEntropyLoss()

# 옵티마이저
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 훈련
EPOCHS = 500
for epoch in range(EPOCHS + 1):
    for batch_idx, samples in enumerate(dataLoader):
        
        x_train, y_train = samples

        # 순전파
        prediction = model(x_train)

        # 역전파
        loss = cost(prediction, y_train)
        optimizer.zero_grad() # gradient reset
        loss.backward() # gradient update
        optimizer.step() # W,b update

        if epoch % 10 == 0:
            print('Epoch {:4d}/{} Batch {}/{} Cost {:.9f}'.format(epoch, EPOCHS, batch_idx+1, len(dataLoader), loss.item()))

# 모델 저장
# torch.save(obj=model, f='Softmax_Regression_Model.pt')
# 모델 불러오기
# loaded_model = torch.load(f='Softmax_Regression_Model.pt')