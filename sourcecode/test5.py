import numpy as np
import matplotlib as plt

train = np.loadtxt("click.csv",dtype="int",delimiter=",",skiprows=1)
trainX = train[:,0]
trainY = train[:,1]
print(trainX,trainY)

mu = trainX.mean()
sigma = trainX.std()
print(mu,sigma)

def standardize(x):
    return (x-mu)/sigma
trainZ = standardize(trainX)

theta0 = np.random.rand()
theta1 = np.random.rand()

def f(x):
    return theta0 + theta1*x

def E(x,y):
    return 0.5*np.sum((y-f(x))**2)

error = E(trainX,trainZ)
print(error)
ETA = 1e-3
diff = 1
count = 0
while diff > 1e-2:
    new_theta0 = theta0 + ETA*np.sum(f(trainZ)-trainY)
    new_theta1 = theta1 + ETA*np.sum((f(trainZ)-trainY)*trainX)

    theta0 = new_theta0
    theta1 = new_theta1

    diff = error - E(trainX,trainZ)
    print(diff)

    count += 1
    log = '{}回目: theta0 = {:.3f}, theta1 = {:.3f},差分 = {:.4f}'
    print(log.format(count,theta0,theta1,diff))

x = np.linspace(-3, 4, 100)
plt.plot(trainX, trainY,'o')
plt.plot(x,f(x))
plt.show()