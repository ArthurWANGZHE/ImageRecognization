import pandas as pd
import numpy as np
import time

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
answer = []
for i in range(28000):
    answer.append(df_answer[i][1])


def caculate_distance(test, train):
    dist = 0
    a = 0
    for i in range(783):
        s = (float(test[i]) - float(train[i + 1])) ** 2
        #   print(s)
        a += 1
        dist += s
    distance_ = (dist/783) ** 0.5
    return distance_

print("stage1")
caculate_distance(df_test[0], df_train[0])
# print(df_train[0][0])
# print(df_train[0])

labeltest = []
for i in range(10):
    distance = []
    labelTrain = []
    for j in range(42000):
        distance.append(caculate_distance(df_test[i], df_train[j]))
        labelTrain.append(df_train[j][0])
    distance2 = sorted(distance)
    ad = []
    ar = []
    for p in range(3):
        a = distance2[p]
        b = distance.index(a)
        ad.append(a)
        ar.append(df_train[b][0])
    S = 0
    S_ = 0
    for i in range(1):
        a_ = ad[i] * ar[i]
        S = S + a_
#        print(S)
        S_ = S_ + ad[i]
 #       print(S_)
    labelType = S / S_
    if labelType <= 0.5:
        labeltest.append(0)
    elif labelType <= 1.5 and labelType > 0.5:
        labeltest.append(1)
    elif labelType <= 2.5 and labelType > 1.5:
        labeltest.append(2)
    elif labelType <= 3.5 and labelType > 2.5:
        labeltest.append(3)
    elif labelType <= 4.5 and labelType > 3.5:
        labeltest.append(4)
    elif labelType <= 5.5 and labelType > 4.5:
        labeltest.append(5)
    elif labelType <= 6.5 and labelType > 5.5:
        labeltest.append(6)
    elif labelType <= 7.5 and labelType > 6.5:
        labeltest.append(7)
    elif labelType <= 8.5 and labelType > 7.5:
        labeltest.append(8)
    elif labelType <= 9.5 and labelType > 8.5:
        labeltest.append(9)
# print(labeltest)
# print("stage2")
"""
correct = 0
for i in range(28000):
    if labeltest[i] == answer[i]:
        correct += 1
"""
print(labeltest)
print(answer)
#print(correct / 28000)
print("finish")
T2 = time.time()
print('程序运行时间:%s秒' % ((T2 - T1)))