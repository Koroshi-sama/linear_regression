import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


data = pd.read_csv('data.csv')

def gradient_descent(theta1, theta0, data, learning_rate):
    temp_theta0 = 0
    temp_theta1 = 0

    m = len(data)
    for i in range(m):
        x = (data.iloc[i].km - data.km.min()) / (data.km.max() - data.km.min())
        y = (data.iloc[i].price - data.price.min()) / (data.price.max() - data.price.min())

        temp_theta1 +=  x * ((theta1 * x + theta0) - y)
        temp_theta0 += ((theta1 * x + theta0) - y)
    
    r_theta1 = theta1 - (1/m * learning_rate * temp_theta1)
    r_theta0 = theta0 - (1/m * learning_rate * temp_theta0)

    return r_theta1, r_theta0

theta1 = 0
theta0 = 0
learning_rate = 0.1
epochs = 500

for _ in range(epochs):
    theta1, theta0 = gradient_descent(theta1, theta0, data, learning_rate)

print(theta1, theta0)

x_range = sorted(data.km)

pre_km = 61789
predicted_price = (theta1 * ((pre_km - data.km.min()) / (data.km.max() - data.km.min())) + theta0) * (data.price.max() - data.price.min()) + data.price.min()
print(predicted_price)

plt.scatter(data.km, data.price, color='red', alpha=0.5)
plt.plot(x_range, [(theta1 * ((x - data.km.min()) / (data.km.max() - data.km.min())) + theta0) * (data.price.max() - data.price.min()) + data.price.min() for x in x_range], color='blue')

plt.show()

