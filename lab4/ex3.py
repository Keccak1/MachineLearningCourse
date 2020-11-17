# using real data, optimize classifier to predict given values

# split dataset into a training set and a test set
# train model on the training set
# calculate TP, FP, TN, FN on test set
# calculate sensitivity, specificity, positive predictivity and negative predictivity


from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

data = pd.read_csv('./data.txt')
data = data.values


X = np.ones((data.shape[1], data.shape[0])) 
X[1:3, :] = data[:, 0:2].T
y = data[:, 2:3].T
X[1, :] = (X[1, :] - np.std(X[1, :])) / np.mean(X[1, :])
X[2, :] = (X[2, :] - np.std(X[2, :])) / np.mean(X[2, :])
theta = np.zeros((X.shape[0], 1))

hypothesis_f = lambda theta, X: 1/ (1+np.exp(-1*theta.T@X))
crossentropy_f = lambda y, h_x: 1/ -y*np.log(h_x+0.00001) - (1-y)*np.log(1-h_x+0.00001)
cost_f = lambda crossentropy, X:  np.sum(crossentropy, axis=1) / X.shape[1]
theta_derivs_f = lambda h_x, y, X: sum((h_x-y) @ X.T) / X.shape[1] 


max_iter = 10000
eps = 0.00001
alpha = 0.05
prev_cost = 99

for i in range(max_iter):
    h_x = hypothesis_f(theta, X)
    crossentropy = crossentropy_f(y, h_x)
    cost = cost_f(crossentropy, X)
    theta_derivs = theta_derivs_f(h_x, y, X)
    theta_derivs.shape = [len(theta_derivs), 1]
    theta = theta - alpha*theta_derivs

    print(f"epoch: {i+1}, cost: {cost}")

    if np.abs(prev_cost - cost) < eps:
        break

    prev_cost = cost

print(theta)

x1 = np.linspace(np.min(X[1, :]), np.max(X[1, :]), 100)
x2 = -theta[0, 0]/theta[2, 0] - theta[1, 0]/theta[2, 0] * x1

X_positive = X[:, y[0, :] == 1]
X_negative = X[:, y[0, :] == 0]

plt.plot(X_positive[2, :], X_positive[1, :], '+')
plt.plot(X_negative[2, :], X_negative[1, :], 'o')
plt.plot(x2, x1, '-')
plt.show()