## Getting started


```python
from __future__ import print_function
import torch
```

Construct a 5x3 matrix, uninitialized:


```python
x = torch.empty(5,3)
print(x)
```

    tensor([[ 0.0000e+00,  0.0000e+00,  1.5489e+18],
            [-4.6577e-10,  1.1210e-44,  0.0000e+00],
            [ 0.0000e+00,  0.0000e+00,  0.0000e+00],
            [ 0.0000e+00,  0.0000e+00,  0.0000e+00],
            [ 0.0000e+00,  0.0000e+00,  0.0000e+00]])


Construct a randomly initialzied matrix:


```python
x = torch.rand(5,3)
print(x)
```

    tensor([[0.5325, 0.0772, 0.1911],
            [0.4686, 0.2347, 0.0857],
            [0.0931, 0.5222, 0.8261],
            [0.7747, 0.3430, 0.2838],
            [0.6866, 0.8283, 0.5899]])


Construct a matrix filled zeros and of dtype long:


```python
x = torch.zeros(5,3,dtype=torch.long)
print(x)
```

    tensor([[0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]])


Construct a tensor directly from data:


```python
x = torch.tensor([5.5, 3])
print(x)
```

    tensor([5.5000, 3.0000])


create a tensor based on an existing tensor. By default, the returned Tensor has the same torch.dtype and torch.device as this tensor


```python
x = x.new_ones(5,3)
# x = x.new_ones(5,3,dtype=torch.double)
print(x)
```

    tensor([[1., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.]])



```python
x = torch.randn_like(x, dtype=torch.float)
print(x)
```

    tensor([[ 2.4408, -2.1528, -1.0814],
            [ 0.9136, -0.8565, -1.7178],
            [-0.6645,  2.1785, -0.2689],
            [-0.0453, -0.9448,  0.4829],
            [-0.2538, -1.6666, -0.0819]])



```python
y = torch.rand(5,3)
print(x+y)
print(torch.add(x,y))
```

    tensor([[ 3.2664e+00, -1.2895e+00, -8.7339e-01],
            [ 1.3830e+00, -3.8670e-01, -1.3328e+00],
            [ 2.4645e-03,  2.6291e+00,  3.8403e-01],
            [ 7.8488e-01, -7.0143e-04,  1.2973e+00],
            [-2.4271e-01, -1.3265e+00,  8.4172e-01]])
    tensor([[ 3.2664e+00, -1.2895e+00, -8.7339e-01],
            [ 1.3830e+00, -3.8670e-01, -1.3328e+00],
            [ 2.4645e-03,  2.6291e+00,  3.8403e-01],
            [ 7.8488e-01, -7.0143e-04,  1.2973e+00],
            [-2.4271e-01, -1.3265e+00,  8.4172e-01]])



```python
result = torch.empty(5, 3)
torch.add(x, y, out=result)
print(result)
```

    tensor([[ 3.2664e+00, -1.2895e+00, -8.7339e-01],
            [ 1.3830e+00, -3.8670e-01, -1.3328e+00],
            [ 2.4645e-03,  2.6291e+00,  3.8403e-01],
            [ 7.8488e-01, -7.0143e-04,  1.2973e+00],
            [-2.4271e-01, -1.3265e+00,  8.4172e-01]])



```python
# adds x to y
y.add_(x)
print(y)
```

    tensor([[ 3.2664e+00, -1.2895e+00, -8.7339e-01],
            [ 1.3830e+00, -3.8670e-01, -1.3328e+00],
            [ 2.4645e-03,  2.6291e+00,  3.8403e-01],
            [ 7.8488e-01, -7.0143e-04,  1.2973e+00],
            [-2.4271e-01, -1.3265e+00,  8.4172e-01]])


Any operation that mutates a tensor in-place is post-fixed with an _. For example: x.copy_(y), x.t_(), will change x.


```python
print(x[:,1])
```

    tensor([-2.1528, -0.8565,  2.1785, -0.9448, -1.6666])


Resizing: if you want to resize/reshape tensor, you can use torch.view


```python
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
print(x)
print(y)
print(z)
```

    tensor([[ 1.0306, -1.1114,  1.1970, -0.9531],
            [-1.3080, -1.1039, -2.2740,  1.2042],
            [-1.4187, -0.7572, -0.1196, -1.2476],
            [-1.3031,  1.3704,  0.2352,  1.5696]])
    tensor([ 1.0306, -1.1114,  1.1970, -0.9531, -1.3080, -1.1039, -2.2740,  1.2042,
            -1.4187, -0.7572, -0.1196, -1.2476, -1.3031,  1.3704,  0.2352,  1.5696])
    tensor([[ 1.0306, -1.1114,  1.1970, -0.9531, -1.3080, -1.1039, -2.2740,  1.2042],
            [-1.4187, -0.7572, -0.1196, -1.2476, -1.3031,  1.3704,  0.2352,  1.5696]])



```python
x = torch.randn(1)
print(x)
print(x.item())
```

    tensor([0.4732])
    0.47318047285079956


## NumPy Bridge

The Torch Tensor and NumPy array will share their underlying memory locations (if the Torch Tensor is on CPU), and changing one will change the other.


```python
a = torch.ones(5)
print(a)
```

    tensor([1., 1., 1., 1., 1.])



```python
b = a.numpy()
print(b)
```

    [1. 1. 1. 1. 1.]



```python
# a.add_(1)
a.add_(1)
print(a)
print(b)
```

    tensor([2., 2., 2., 2., 2.])
    [2. 2. 2. 2. 2.]


### Converting NumPy Array to Torch Tensor


```python
import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a,1,out=a)
print(a)
print(b)
```

    [2. 2. 2. 2. 2.]
    tensor([2., 2., 2., 2., 2.], dtype=torch.float64)

<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

<p>
  When \(a \ne 0\), there are two solutions to \(ax^2 + bx + c = 0\) and they are
  \[x = {-b \pm \sqrt{b^2-4ac} \over 2a}.\]
</p>