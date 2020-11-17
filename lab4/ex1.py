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

X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 25], dtype=np.float32)
y = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1,  1,  1], dtype=np.float32)

theta = np.array([0, 0], dtype=np.float32)

# optimization loop

hypothesis_f = lambda theta, X: 1/ (1+np.exp(-theta[0] - theta[1]*X))
crossentropy_f = lambda y, h_x: -y*np.log(h_x + 0.00001) - (1-y)*np.log(1-h_x + 0.00001)
cost_f = lambda crossentropy, size:  sum(crossentropy)/size
theta0_deriv_f = lambda h_x, y: sum(h_x-y)/len(y)
theta1_deriv_f = lambda h_x, y: sum((h_x-y)*X)/len(y)

max_iter = 5000
eps = 0.00001
prev_cost = 99
alpha = 0.1

for i in range(max_iter):

    h_x =hypothesis_f(theta, X)
    crossentropy = crossentropy_f(y,h_x)
    cost = cost_f(crossentropy, len(y))
    t0_d = theta0_deriv_f(h_x, y)
    t1_d = theta1_deriv_f(h_x, y)

    theta[0] -= alpha*t0_d
    theta[1] -= alpha*t1_d

    print(f"epoch: {i+1}, cost: {cost}")

    if np.abs(prev_cost - cost) < eps:
        break

    prev_cost = cost

print(theta)

x_samples = np.linspace(min(X), max(X), 100)
y_samples = 1 / (1 + np.exp(-theta[0] - theta[1]*x_samples))

plt.plot(X, y, 'o')
plt.plot(x_samples, y_samples, '-')
plt.show()



