# fit the sigmoid curve and calculate decision boundary using given dataset

# a cheat sheet:
# in an optimization loop
# first calculate hypothesis for each datapoint x in X: h = 1 / (1 + exp(-theta0-theta1*x))
# then calculate crossentropy: -y*log(h) - (1-y)*log(1-h)
# and cost: sum(crossentropy) / len(x)
# next calculate derivatives for theta 0 and theta1 (similar to those in linear regression)
# theta0_deriv = sum(h - y) / len(y), theta1_deriv = sum((h-y)*X)
# and then update theta weights
# theta = theta - lr*theta_deriv

# check if cost is getting lower through iterations
# if not, try to modify the learning rate

# calculating decision boundary might look like this:
# theta[0] + theta[1]*x = 0
# theta[1]*x = -theta[0]
# x = -theta[0]/theta[1]

# the result might look like below

from matplotlib import pyplot as plt
import numpy as np

X = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1],
              [1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 25]], dtype=np.float32)
y = np.array([[0, 0, 0, 0, 0, 0, 1, 1, 1,  1,  1]], dtype=np.float32)

theta = np.array([[0], [0]], dtype=np.float32)

# optimization loop

hypothesis_f = lambda theta, X: 1/ (1+np.exp(-1*theta.T@X))
crossentropy_f = lambda y, h_x: 1/ -y*np.log(h_x+0.00001) - (1-y)*np.log(1-h_x+0.00001)
cost_f = lambda crossentropy, X:  np.sum(crossentropy, axis=1) / X.shape[1]
theta_derivs_f = lambda h_x, y, X: sum((h_x-y) @ X.T) / X.shape[1] 

max_iter = 10000
eps = 0.00001
alpha = 0.1
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

x_samples = np.ones((X.shape[0], 100))
x_tmp = np.linspace(min(X[1,:]), max(X[1,:]), 100)
x_samples[1, :] = x_tmp
y_samples = 1 / (1 + np.exp(-theta.T@x_samples))

plt.plot(X[1,:], y[0, :], 'o')
plt.plot(x_tmp, y_samples[0, :], '-')
plt.show()