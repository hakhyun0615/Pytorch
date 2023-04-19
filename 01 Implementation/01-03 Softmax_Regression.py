import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# GPU
# USE_CUDA = torch.cuda.is_available() 
# device = torch.device("cuda" if USE_CUDA else "cpu")
# print("{}로 학습합니다.".format(device))

# 랜덤 시드
torch.manual_seed(1)

# 데이터
x_train = torch.FloatTensor([[1, 2, 1, 1],
                            [2, 1, 3, 2],
                            [3, 1, 3, 4],
                            [4, 1, 5, 5],
                            [1, 7, 5, 5],
                            [1, 2, 5, 6],
                            [1, 6, 6, 6],
                            [1, 7, 7, 7]])
y_train = torch.LongTensor([2, 2, 2, 1, 1, 1, 0, 0])

# 데이터셋, 데이터로더
dataset = TensorDataset(x_train, y_train)
dataLoader = DataLoader(dataset, batch_size=2 ,shuffle=True, drop_last=True)

# 모델
class SoftmaxRegressionModel(nn.Module):
    def __init__(self): 
        super(SoftmaxRegressionModel, self).__init__()
        self.linear = nn.Linear(4, 3)

    def forward(self, x):
        return self.linear(x)
    
model = SoftmaxRegressionModel()

# 비용함수
cost = nn.CrossEntropyLoss()

# 옵티마이저
optimizer = optim.SGD(model.parameters(), lr=0.01) 

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
            print('Epoch {:4d}/{} Batch {}/{} Cost {:.6f}'.format(epoch, EPOCHS, batch_idx+1, len(dataLoader), loss.item()))

# 모델 저장
# torch.save(obj=model, f='Softmax_Regression_Model.pt')
# 모델 불러오기
# loaded_model = torch.load(f='Softmax_Regression_Model.pt')