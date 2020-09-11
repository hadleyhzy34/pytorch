from __future__ import print_function
import torch

x = torch.ones(2,2,requires_grad=True)
print(x)

y = x + 2
print(y)

#scalar cannot show calculated backward unless using tensor as additional input
i = torch.ones(2,2)
y.backward(i)
print(x.grad)
print(x)