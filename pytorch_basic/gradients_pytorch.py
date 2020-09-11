# 1) design model(input, output size, forward pass)
# 2) construct loss and optimizer
# 3) training loop
#   -forward pass: compute prediction
#   -backward pass: gradients
#   -update weights



import torch
import torch.nn as nn

##Gradient Descent from scratch
X = torch.tensor([1,2,3,4], dtype=torch.float32)
Y = torch.tensor([2,4,6,8], dtype=torch.float32)

w = torch.tensor(0.0, dtype= torch.float32, requires_grad=True)

#model prediction

def forward(x):
    return w * x

# #loss
# def loss(y, y_predicted):
#     return ((y_predicted-y)**2).mean()

#gradient
#MSE mean square error: 1/N*(w*x-y)**2
#dJ/dw = 1/N *2*x(w*x-y)
def gradient(x,y,y_predicted):
    loss(y,y_predicted).backward()
    return w.grad


print(f'prediction before training: f(5) = {forward(5):.3f}')

#training
learning_rate = 0.01
n_iters = 10

loss = nn.MSELoss()
optimizer = torch.optim.SGD([w], lr=learning_rate)

for epoch in range(n_iters):
    #prediction = forward pass
    y_pred = forward(X)

    #loss
    l = loss(Y, y_pred)

    #gradients
    l.backward() #dl/dw
    # dw = gradient(X,Y,y_pred)

    # #update wrights
    # with torch.no_grad():
    #     #below operation needs not to be in gradient tracking part
    #     w -= learning_rate * w.grad

    #update weights
    optimizer.step()

    #zero gradients, w.grad will be accumulated, needs to be set to be zero before next iteration
    # w.grad.zero_()
    optimizer.zero_grad()

    if epoch % 1 ==0:
        print(f'epoch {epoch+1}: w={w:.3f}, loss = {l:.8f}')


print(f'prediction after training: f(5) = {forward(5):.3f}')