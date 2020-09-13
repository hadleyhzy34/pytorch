import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# 0)prepare data
## dataset from kaggle:https://www.kaggle.com/quantbruce/real-estate-price-prediction
real_estate = pd.read_csv('resources/real_estate.csv')
#summary of real_estate data
# print(real_estate.head())
real_estate = real_estate.drop(['No'],axis=1)
print(real_estate.shape)
print(real_estate.info())

print(real_estate.isnull().sum())

# real_estate = (real_estate - real_estate.mean())/(real_estate.max()-real_estate.min())
# print(real_estate.head)
## Data splitting to training and testing model
data_train, data_test = train_test_split(real_estate, train_size=0.70, test_size = 0.30, random_state=100)

print(data_train.shape, data_test.shape)

## split data for input and output
y_train = data_train.pop('Y house price of unit area')
x_train = data_train
x_train = (x_train - x_train.mean())/(x_train.max()-x_train.min())

y_test = data_test.pop('Y house price of unit area')
x_test = data_test
x_test = (x_test - x_test.mean())/(x_test.max()-x_test.min())


# print(x_train.head(), x_train.shape)
# print(y_train.head(), y_train.shape)

##weights and bias initialization
##y=w1*x1+w2*x2...wn*xn+b
input_size = x_train.shape[1]
w = np.zeros((input_size,))
b = 0

#model prediction: forward pass
def forward(x):
    return np.dot(x, w) + b

#loss function based on mean square error
def loss(y, y_predicted):
    # print(y_predicted.T.shape, y.T.shape)
    square = np.square(y_predicted.T-y.T)
    # print(square.shape)
    # print(square.dtype)
    return square.mean()*(1/2)

#gradient
#MSE mean square error: (1/N)*(w*x+b-y)**2
#dJ/dw(i) = (1/N) *2*x(w(i)*x+b-y)

## gradient of weight
def gradient_weight(x,y,y_predicted):
    dw = (1/input_size)*np.dot(x.T, y_predicted-y)
    return dw;

##gradient of bias
def gradient_bias(x,y,y_predicted):
    db = (1/input_size)*np.sum(y_predicted-y)
    return db;

## training 
learning_rate = 0.01
n_iters = 100

plt_x = []
plt_y1 = []
plt_y2 = []

for epoch in range(n_iters):
    #prediction = forward pass
    y_pred = forward(x_train)

    #loss
    l = loss(y_train, y_pred)
    # print(l)
    # print(l.dtype)

    #gradients
    dw = gradient_weight(x_train, y_train, y_pred)
    db = gradient_bias(x_train, y_train, y_pred)

    #update wrights
    w -= learning_rate * dw
    b -= learning_rate * db

    # if epoch % 1 ==0:
    #     print(f'epoch {epoch+1}: w={w}, loss = {l:.4f}')
    
    ##prepare plot data
    plt_x.append(epoch)
    plt_y1.append(l)

    y_test_p = forward(x_test)
    l_p = loss(y_test, y_test_p)
    plt_y2.append(l_p)

##loss curve
plt.plot(plt_x, plt_y1, 'b')
plt.plot(plt_x, plt_y2, 'r')
plt.show()

##prediciton
y_test_pred = forward(x_test)
result = r2_score(y_true=y_test,y_pred=y_test_pred)
print(result)

# plt.plot(y_test_pred.T, 'g')
# plt.plot(y_test, 'r')
# plt.show()
print(x_test.iloc[:,1])
print(x_test[0:1].shape)
print(y_test.T.shape)

##plot to check if those features accuracy for final reuslts
plt.plot(data_test.iloc[:,5], y_test, 'ro')
plt.plot(data_test.iloc[:,5], y_test_pred, 'bo')
plt.plot((data_test.iloc[:,5],data_test.iloc[:,5]), (y_test,y_test_pred), c='black')
plt.show()