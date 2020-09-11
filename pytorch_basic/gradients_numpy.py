import torch
import numpy as np

# ##simple example using backpropagation
# x = torch.tensor(1.0)
# y = torch.tensor(2.0)

# w = torch.tensor(1.0, requires_grad=True)

# #forward pass and compute the loss
# y_hat = w*x
# loss = (y_hat - y)**2

# print(loss)

# #backward pass
# loss.backward()
# print(w.grad)

##Gradient Descent from scratch
X = np.array([1,2,3,4], dtype=np.float32)

Y = np.array([2,4,6,8], dtype=np.float32)

w = 0.0

#model prediction

def forward(x):
    return w * x

#loss
def loss(y, y_predicted):
    return ((y_predicted-y)**2).mean()

#gradient
#MSE mean square error: 1/N*(w*x-y)**2
#dJ/dw = 1/N *2*x(w*x-y)
def gradient(x,y,y_predicted):
    return np.dot(2*x, y_predicted-y).mean()


print(f'prediction before training: f(5) = {forward(5):.3f}')

#training
learning_rate = 0.01
n_iters = 10

for epoch in range(n_iters):
    #prediction = forward pass
    y_pred = forward(X)

    #loss
    l = loss(Y, y_pred)

    #gradients
    dw = gradient(X,Y,y_pred)

    #update wrights
    w -= learning_rate * dw

    if epoch % 1 ==0:
        print(f'epoch {epoch+1}: w={w:.3f}, loss = {l:.8f}')