import numpy as np
import matplotlib as plt

train = np.loadtxt("click.csv",dtype="int",delimiter=",",skiprows=1)
trainX = train[:,0]
trainY = train[:,1]
print(trainX,trainY)

mu = trainX.mean()
sigma = trainX.std()
print(mu,sigma)


