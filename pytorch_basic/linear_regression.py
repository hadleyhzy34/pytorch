import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

# 0)prepare data
# https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_regression.html
#Xarray of shape [n_samples, n_features] 
#yarray of shape [n_samples] or [n_samples, n_targets]
X_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=50,random_state=1)

# linear model requires scalar to be float type other than double type
X=torch.from_numpy(X_numpy.astype(np.float32))
y=torch.from_numpy(y_numpy.astype(np.float32))
# X=torch.from_numpy(X_numpy)
# y=torch.from_numpy(y_numpy)

# print(X.dtype)
# print(X.shape)
# print(y.shape)

#resize tensor to be 100,1
y = y.view(y.shape[0],1)

n_samples, n_features = X.shape

#1)model
input_size = n_features
output_size =1
model = nn.Linear(input_size, output_size)

#2) loss and optimizer
learning_rate = 0.1
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

#3) training loop
num_epochs = 100
for epoch in range(num_epochs):
    #forward pass and loss
    y_predicted = model(X)
    loss = criterion(y_predicted, y)

    #backward pass
    loss.backward()

    #update
    optimizer.step()
    optimizer.zero_grad()

    if(epoch+1)%10 == 0:
        print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')
    
#plot
##Can't call numpy() on Tensor that requires grad
# predicted = model(X).numpy()
# ##cannot plot tensor
# predicted = model(X)
predicted = model(X).detach().numpy()
plt.plot(X_numpy, y_numpy, 'ro')
plt.plot(X_numpy, predicted, 'b')
plt.show()