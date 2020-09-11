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

# class test_base():
#     def __init__(self,x):
#         self.x = x
#         return self.x

# class test(): 
#     def __init__(self,x):
#         pass

#     def func(self, x):
#         return x

# test1 = test()
# print(test1.func(5))
# print(test1(5))
