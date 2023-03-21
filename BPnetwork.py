import numpy as np
import pandas as pd
from sklearn import preprocessing
import torch
import torch.nn.functional as Fun
import time

# 指定gpu
device = torch.device("cuda:0"if torch.cuda.is_available()else "cpu")

# 参数设置
lr = 0.02
epochs = 3000
n_feature = 783
n_hidden = 20
n_output = 10

# 准备数据集
T1 = time.time()
df_test = pd.read_csv('test.csv')
# 28000 rows * 784 columns (783个变量，1个序号）
df_train = pd.read_csv('train.csv')
# 42000 rows * 785 columns (783个变量，1个标签，1个序号）
df_answer = pd.read_csv('标准答案.csv')
# print(df_train)
df_test = np.array(df_test)
df_train = np.array(df_train)
df_answer = np.array(df_answer)
# 删除train 标签
df_train0 = np.delete(df_train, 0, axis=1)
df_train0 = np.delete(df_train0, 0, axis=1)
df_test = np.delete(df_test, 0, axis=1)
# 获取train 标签
df_train1 = df_train[:, 0]
answer = []
for i in range(28000):
    answer.append(df_answer[i][1])

# 数据预处理
min_max_scaler = preprocessing.MinMaxScaler()
df_train = min_max_scaler.fit_transform(df_train0)
df_test = min_max_scaler.fit_transform(df_test)

x_train = torch.FloatTensor(df_train).to(device)
x_train_answer = torch.LongTensor(df_train1).to(device)
x_test = torch.FloatTensor(df_test).to(device)
x_test_answer = torch.FloatTensor(df_answer).to(device)


class BPnetwork(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(BPnetwork, self).__init__()
        self.hiddden = torch.nn.Linear(n_feature, n_hidden)
        self.out = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = Fun.relu(self.hiddden(x))
        out = Fun.softmax(self.out(x), dim=1)
        return out


net = BPnetwork(n_feature=n_feature, n_hidden=n_hidden, n_output=n_output).to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
loss_fun = torch.nn.CrossEntropyLoss()

# training
loss_steps = np.zeros(epochs)
accuracy_steps = np.zeros(epochs)

for epoch in range(epochs):
    train_pred = net(x_train).to(device)
    loss = loss_fun(train_pred, x_train_answer)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    loss_steps[epoch] = loss.item()
    running_loss = loss.item()
    print(f"第{epoch}次训练，loss={running_loss}".format(epoch, running_loss))
    with torch.no_grad():
        test_pred = net(x_test).to(device)
        c= torch.argmax(test_pred, dim=1)

answer_=c.tolist()
correct = 0
for i in range(len(answer)):
    if answer_[i]==answer[i]:
        correct +=1
print(len(answer),len(answer_))
print(correct/len(answer))
T2 = time.time()
print('程序运行时间:%s秒' % ((T2 - T1)))
