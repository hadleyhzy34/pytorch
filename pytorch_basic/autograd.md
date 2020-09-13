```python
import torch
```

Create a tensor and set requires_grad = True to track computation with it


```python
x = torch.ones(2,2,requires_grad=True)
print(x)
```

    tensor([[1., 1.],
            [1., 1.]], requires_grad=True)


Do a tensor operation


```python
y = x + 2
print(y)
t = 3*x
print(t)
#scalar cannot show calculated backward unless using tensor as additional input
#∂y/∂xi=1 when xi = any value
i = torch.ones(2,2)
y.backward(i)
#creating another gradient function with repect to x, final grad result will be sum of y and t
x.grad.zero_() #this will erase current output of gradients
t.backward(i)
print(x.grad)
```

    tensor([[3., 3.],
            [3., 3.]], grad_fn=<AddBackward0>)
    tensor([[3., 3.],
            [3., 3.]], grad_fn=<MulBackward0>)
    tensor([[3., 3.],
            [3., 3.]])


Each tensor has a .grad_fn attribute that references a Function that has created the Tensor (except for Tensors created by the user - their grad_fn is None).


```python
print(x.grad_fn)
print(y.grad_fn)
```

    None
    <AddBackward0 object at 0x1107f4100>



```python
z = y*y*3
out = z.mean()
print(z,out)
```

    tensor([[27., 27.],
            [27., 27.]], grad_fn=<MulBackward0>) tensor(27., grad_fn=<MeanBackward0>)


.requires_grad_( ... ) changes an existing Tensor’s requires_grad flag in-place. The input flag defaults to False if not given.


```python
a = torch.randn(2, 2)
a = ((a * 3) / (a - 1))
print(a.requires_grad)
a.requires_grad_(True)
print(a.requires_grad)
b = (a * a).sum()
print(b)
print(b.grad_fn)
```

    False
    True
    tensor(383235.5312, grad_fn=<SumBackward0>)
    <SumBackward0 object at 0x1103b1280>


### Gradients


```python
print(out)
out.backward()
print(x.grad)
```

    tensor(27., grad_fn=<MeanBackward0>)
    tensor([[7.5000, 7.5000],
            [7.5000, 7.5000]])


You should have got a matrix of 4.5. Let’s call the out Tensor “o”. We have that o=14∑izi, zi=3(xi+2)2 and zi∣∣xi=1=27. Therefore, ∂o/∂xi=3/2(xi+2), hence ∂o/∂xi∣∣xi=1=92=4.5.


```python
x = torch.randn(3,requires_grad=True)
print(x)
y = x*2
print(y)
#data.norm() equals to euclidean norm
print(y[0]*y[0]+y[1]*y[1]+y[2]*y[2])
print(y.data.norm())
while y.data.norm() < 100:
    y = y * 2

print(y)
v = torch.ones(3)
y.backward(v)

print(x.grad)
```

    tensor([-0.7527, -1.2910,  1.2412], requires_grad=True)
    tensor([-1.5055, -2.5820,  2.4825], grad_fn=<MulBackward0>)
    tensor(15.0960, grad_fn=<AddBackward0>)
    tensor(3.8854)
    tensor([-48.1744, -82.6248,  79.4394], grad_fn=<MulBackward0>)
    tensor([64., 64., 64.])
    True
    True
    False



```python
print(x.requires_grad)
print((x ** 2).requires_grad)

with torch.no_grad():
    print((x ** 2).requires_grad)
```

    True
    True
    False



```python
print(x.requires_grad)
y = x.detach()
print(y.requires_grad)
#True if two tensors have the same size and elements, False otherwise.
print(x.eq(y).all())
print(x)
print(y)
```

    True
    False
    tensor(True)
    tensor([-0.7527, -1.2910,  1.2412], requires_grad=True)
    tensor([-0.7527, -1.2910,  1.2412])



```python

```
