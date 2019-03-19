import numpy as np
import matplotlib.pyplot as plt

# 学習データを読み込む
train = np.loadtxt("click2.csv", dtype="int", delimiter=",", skiprows=1)
train_x = train[:, 0]
train_y = train[:, 1]

# 標準化
mu = train_x.mean()
sigma = train_x.std()


def standardize(x):
    return (x - mu) / sigma


train_z = standardize(train_x)

# パラメータを初期化
# theta = np.random.rand(3)
theta = [1, 2, 3]


def to_matrix(x):
    return np.vstack([np.ones(x.size), x, x ** 2]).T





def f(x):
    return np.dot(theta, x)


X = to_matrix(train_x)

print(X)
print(theta)
y=f(X)
# print(f(X))

#
# import numpy as np
#
# train = np.loadtxt('click2.csv', delimiter=',', dtype='int', skiprows=1)
# train_x = train[:, 0]
# train_y = train[:, 1]
# print(train_x)
# print(train_y)
#
# mu = train_x.mean()
# sigma = train_y.std()
#
#
# def standardize(x):
#     return (x - mu) / sigma
#
#
# train_z = standardize(train_x)
# print(train_z)
#
# theta = np.random.rand(3)
#
# print(theta)
#
#
# def to_matrix(x):
#     return np.vstack([np.ones(x.size), x, x ** 2]).T
#
#
# test = to_matrix(train_x)
# print(test)
# X = to_matrix(train_z)
#
# def f(x):
#     return np.dot(x,theta)
#
# print(f(test))
#
# def E(x,y):
#     return 0.5*np.sum((y-f(x))**2)
