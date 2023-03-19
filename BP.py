import numpy as np
import pandas as pd
from time import time


# 特征缩放类(归一法
class Feature_scaling:
    def __init__(self, train):
        self.min = np.min(train)
        self.ptp = np.ptp(train)

    def __str__(self):
        return f"avr = {self.avr}, ptp = {self.ptp}"

    def initial_feature(self, feature):
        m = feature.shape[0]
        a0 = np.ones(m).reshape(m, 1)
        return np.hstack((a0, np.divide(np.subtract(feature, self.min), self.ptp)))

    def new_data_updating(self, test):
        return np.divide(np.subtract(test, self.min), self.ptp)


class DATA:
    trainX = None
    trainLabel = None
    trainM = None
    PixelN = None
    test = None
    Fs = None
    def read_data(self, path, first=True):
        train = pd.read_csv(path)
        X = np.asarray(train.loc[:, train.columns[first:]])  # 特征读取
        if first:
            label = train.loc[:, 'label']
            label = np.eye(10, dtype=np.uint8)[label]  # 获取label的onehot编码矩阵
        if first:
            self.trainM, self.PixelN = X.shape
            print("样本个数和像素数为:",self.trainM,self.PixelN)
            self.Fs = Feature_scaling(X)
            self.trainX = self.Fs.initial_feature(X)
            self.trainLabel = label
        else:
            self.test = self.Fs.new_data_updating(X)


class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, z):
        out = np.reciprocal((1 + np.exp(-z)))
        return out

    def backward(self, a):
        dx = (1.0 - a) * a
        return dx


class BPNN:
    def __init__(self, layerDev, epsilon=1, learingRate=0.1, regParam=0.01):
        self.layerL = len(layerDev)
        self.layerDev = [0] + layerDev
        self.thetaDev = [0, 0] + [np.subtract(np.multiply( \
            np.random.rand(layerDev[l + 1], layerDev[l] + 1), \
            2 * epsilon), epsilon) \
            for l in range(self.layerL - 1)]
        self.dwDev = [0, 0] + [np.zeros((layerDev[l + 1], layerDev[l] + 1)) \
                               for l in range(self.layerL - 1)]
        self.AF = Sigmoid()
        self.learingRate = learingRate
        self.regParam = regParam

    def computing_a(self, X):
        a, m = X, X.shape[0]
        aDev = [0, a]
        for l in range(1, self.layerL):
            z = np.dot(a, self.thetaDev[l + 1].T)
            a = self.AF.forward(z)
            if l != self.layerL - 1:
                a = np.concatenate((np.ones(1), self.AF.forward(z)))
            aDev.append(a.copy())
        return aDev

    def update(self, m):
        for l in range(2, self.layerL + 1):
            self.dwDev[l] /= m
            self.dwDev[l][:, 1:] += self.thetaDev[l][:, 1:] * self.regParam
            self.thetaDev[l] -= self.learingRate * self.dwDev[l]

    def backpropagation(self, X, label):
        aDev = self.computing_a(X)
        # deltaDev = [0,0] + [np.zeros((1,self.layerDev[l]+1)) for l in range(1,self.layerL)]
        deltaDev = list("0" * (self.layerL + 1))
        deltaDev[-1] = aDev[-1] - label
        self.dwDev[-1] += np.matmul(deltaDev[-1].reshape((self.layerDev[-1], 1)), \
                                    aDev[self.layerL - 1].T.reshape(1, self.layerDev[self.layerL - 1] + 1))
        for l in range(self.layerL - 1, 1, -1):
            delta = deltaDev[l + 1]
            if l < self.layerL - 1:
                delta = delta[1:]
            deltaDev[l] = np.dot(self.thetaDev[l + 1].T, delta) * np.concatenate(([1], self.AF.backward(aDev[l][1:])))
            self.dwDev[l] += np.matmul(deltaDev[l][1:].reshape((self.layerDev[l], 1)), \
                                       aDev[l - 1].T.reshape(1, self.layerDev[l - 1] + 1))

    def train(self, X, label):
        print("训练集样本",X.shape)
        m, pixelN = X.shape
        for i in range(m):
            print(i)
            self.backpropagation(X[i], label[i])
        self.update(m)
        for i,theta in enumerate(self.thetaDev):
            print("第i层的权重矩阵为:",theta)


timeStart = time()
Data = DATA()
Data.read_data('train.csv')
BPNN = BPNN(layerDev=[Data.PixelN,3*Data.PixelN//2,200,10])
BPNN.train(Data.trainX,Data.trainLabel)
Data.read_data('test.csv', first=False)
timeEnd = time()
print(timeEnd-timeStart)