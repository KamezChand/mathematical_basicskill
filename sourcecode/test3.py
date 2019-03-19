import numpy as np

train = np.loadtxt("click2.csv", delimiter=',', dtype='int', skiprows=1)
train_x = train[:, 0]
train_y = train[:, 1]
print(train_x, train_y)

# 標準化
mu = train_x.mean()
sigma = train_x.std()


def standardize(x):
    return (x - mu) / sigma


train_z = standardize(train_x)


# print(np.ones(train_x))

def to_matrix(x):
    return np.vstack([np.ones(x.size), x, x ** 2]).T


X = to_matrix(train_x)
print(X)

theta = np.random.rand(3)
print(theta)


def f(x):
    return np.dot(x, theta)


print(f(X))


def E(x, y):
    return 0.5 * (y - f(x)) ** 2

print(E(X,train_y))

theta = theta - ETA*(np.dot((f(X)-train_y),X))