import numpy as np
import matplotlib.pyplot as plt

# 学習データを読み込む
train = np.loadtxt('click2.csv', delimiter=',', dtype='int', skiprows=1)
train_x = train[:, 0]
train_y = train[:, 1]

# 標準化
mu = train_x.mean()
sigma = train_x.std()


def standardize(x):
    return (x - mu) / sigma


train_z = standardize(train_x)

# パラメータを初期化
theta = np.random.rand(3)
theta = [1, 2, 3]


# 学習データの行列を作る
def to_matrix(x):
    return np.vstack([np.ones(x.size), x, x ** 2]).T


X = to_matrix(train_z)


# 予測関数
def f(x):
    return np.dot(x, theta)


print(X)
print(f(X))


def E(x, y):
    return 0.5 * ((y - f(x)) ** 2)

theta = theta + np.dot((y - f(train_x)),train_x)
