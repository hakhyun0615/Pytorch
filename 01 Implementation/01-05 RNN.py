import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# GPU 사용
# USE_CUDA = torch.cuda.is_available() 
# device = torch.device("cuda" if USE_CUDA else "cpu")
# print("{}로 학습합니다.".format(device))

# 데이터
df = pd.read_csv('./stock_data.csv')
x = df.iloc[:,:-1]
y = df.iloc[:,-1:]

# 데이터 전처리
scaler = MinMaxScaler()
x = scaler.fit_transform(x)
y = scaler.fit_transform(y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=False)

# 데이터 텐서 변환
x_train = torch.FloatTensor(x_train)
y_train = torch.FloatTensor(y_train)
x_test = torch.FloatTensor(x_test)
y_test = torch.FloatTensor(y_test)
print("훈련데이터 텐서 변환: ", x_train.shape, y_train.shape)
print("테스트데이터 텐서 변환: ", x_test.shape, y_test.shape)

# 데이터셋, 데이터로더
batch_size = 16
dataset = TensorDataset(x_train, y_train)
dataLoader = DataLoader(dataset, batch_size=batch_size ,shuffle=False)

# 모델
input_size=3
hidden_size=1
output_size=1
class RNN(nn.Module):
    # RNN input (batch_size, time_steps, input_size)
    # RNN output (batch_size, time_steps, hidden_size)
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(hidden_size, output_size), nn.Sigmoid())

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size)
        x, _ = self.rnn(x, h0)
        x = self.fc(x[:,-1])
        return x
  
model = RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
        )

# 비용함수
cost = nn.MSELoss()

# 옵티마이저
LR = 0.01
optimizer = optim.SGD(model.parameters(), lr=LR) 

# 훈련
EPOCHS = 500
loss_history=[]
for epoch in range(EPOCHS + 1):
    loss_batch=0
    for batch_idx, samples in enumerate(dataLoader):
        
        x_train, y_train = samples

        # 순전파
        prediction = model(x_train)

        # 역전파
        loss = cost(prediction, y_train)
        optimizer.zero_grad() # gradient reset
        loss.backward() # gradient update
        optimizer.step() # W,b update

        loss_batch += loss.item()
        if epoch % 10 == 0:
            print('Epoch {:4d}/{} Batch {}/{} Loss {:.6f}'.format(epoch, EPOCHS, batch_idx+1, len(dataLoader), loss.item()))
    loss_history.append(loss_batch/len(dataLoader))
 
# 그래프 1
plt.figure(figsize=(20,10))
plt.plot(loss_history)
plt.show()

# 그래프 2
# def plotting(train_loader, test_loader, actual):
#   with torch.no_grad():
#     train_pred = []
#     test_pred = []

#     for data in train_loader:
#       seq, target = data
#       out = model(seq)
#       train_pred += out.cpu().numpy().tolist()

#     for data in test_loader:
#       seq, target = data
#       out = model(seq)
#       test_pred += out.cpu().numpy().tolist()
      
#   total = train_pred + test_pred
#   plt.figure(figsize=(20,10))
#   plt.plot(np.ones(100)*len(train_pred), np.linspace(0,1,100), '--', linewidth=0.6)
#   plt.plot(actual, '--')
#   plt.plot(total, 'b', linewidth=0.6)

#   plt.legend(['train boundary', 'actual', 'prediction'])
#   plt.show()
# plotting(train_loader, test_loader, df['Close'][sequence_length:])