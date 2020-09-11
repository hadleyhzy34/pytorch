# 1) design model(input, output size, forward pass)
# 2) construct loss and optimizer
# 3) training loop
#   -forward pass: compute prediction
#   -backward pass: gradients
#   -update weights



import torch
import torch.nn as nn

#each row present one sample, each column represent one feature
X = torch.tensor([[1],[2],[3],[4]], dtype=torch.float32)
Y = torch.tensor([[2],[4],[6],[8]], dtype=torch.float32)
X_test = torch.tensor([5], dtype=torch.float32)
n_samples, n_features = X.shape

input_size = n_features
output_size = n_features

#model prediction
# model = nn.Linear(input_size, output_size)

#custom model
class LinearRegression(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        #define layers
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

model = LinearRegression(input_size, output_size)

# #loss
# def loss(y, y_predicted):
#     return ((y_predicted-y)**2).mean()

#gradient
#MSE mean square error: 1/N*(w*x-y)**2
#dJ/dw = 1/N *2*x(w*x-y)
def gradient(x,y,y_predicted):
    loss(y,y_predicted).backward()
    return w.grad

print(model)
print(model(X_test))
#it's quite confusing here that why using model(X_test) instead of model.forward(X_test)
#https://stackoverflow.com/questions/58823011/forward-method-calling-in-linear-class-in-pytorch
#https://github.com/pytorch/pytorch/blob/7073ee209000a7781c0c863c4ef39bb3bfdb4932/torch/nn/modules/module.py#L522
#https://discuss.pytorch.org/t/any-different-between-model-input-and-model-forward-input/3690
#https://stackoverflow.com/questions/111234/what-is-a-callable/115349

#generally speaking, forward is called in the .__call__ function and this 
#function called when the instance is 'called'
print(f'prediction before training: f(5) = {model(X_test).item():.3f}')

#training
learning_rate = 0.01
n_iters = 50

loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(n_iters):
    #prediction = forward pass
    # y_pred = model(X)
    y_pred = model.forward(X)

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
        [w,b] = model.parameters()
        ## w list of list
        print(f'epoch {epoch+1}: w={w[0][0]:.3f}, loss = {l:.8f}')


print(f'prediction after training: f(5) = {model(X_test).item():.3f}')