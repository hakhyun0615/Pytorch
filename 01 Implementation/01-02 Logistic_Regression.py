import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# GPU 사용
# USE_CUDA = torch.cuda.is_available() 
# device = torch.device("cuda" if USE_CUDA else "cpu")
# print("{}로 학습합니다.".format(device))

# 랜덤 시드
torch.manual_seed(1)

# 데이터
x_train = torch.FloatTensor([[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]])
y_train = torch.FloatTensor([[0], [0], [0], [1], [1], [1]])

# 데이터셋, 데이터로더
dataset = TensorDataset(x_train, y_train)
dataLoader = DataLoader(dataset, batch_size=2 ,shuffle=True, drop_last=True)

# 모델
class LogisticRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.linear(x))
    
model = LogisticRegressionModel()

# 비용함수
cost = nn.BCELoss()

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
            correct_prediction = (prediction >= torch.FloatTensor([[0.5]])).float() == y_train
            accuracy = correct_prediction.sum().item() / len(correct_prediction)
            print('Epoch {:4d}/{} Batch {}/{} Cost {:.6f} Accuracy {:2.2f}%'.format(epoch, EPOCHS, batch_idx+1, len(dataLoader), loss.item(), accuracy * 100))

# 모델 저장
# torch.save(obj=model, f='Logistic_Regression_Model.pt')
# 모델 불러오기
# loaded_model = torch.load(f='Logistic_Regression_Model.pt')