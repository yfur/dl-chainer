import numpy as np
from chainer import cuda, Function, gradient_check, Variable, optimizers, \
                    serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L

# x1 = Variable(np.array([1]).astype(np.float32))
# x2 = Variable(np.array([2]).astype(np.float32))
# x3 = Variable(np.array([3]).astype(np.float32))
#
# z = (x1 - 2*x2 - 1)**2 + (x2*x3 - 1)**2 + 1
# print(z.data)
# print()
#
# print(x1.grad)
# print()
# z.backward()
# print(x1.grad)
# print(x2.grad)
# print(x3.grad)
# print()
#
# x = Variable(np.array([-1]).astype(np.float32))
# print(F.sin(x).data, F.sin(x).dtype)
# print(F.sigmoid(x).data, F.sin(x).dtype)
# print()
#
#
# x = Variable(np.array([-0.5]).astype(np.float32))
# z = F.cos(x)
# print(z.data)
# print()
# z.backward()
# print(x.grad)
# print((-1)*F.sin(x))
#
#
# x = Variable(np.array([-1, 0, 1]).astype(np.float32))
# z = F.sin(x)
# z.grad = np.ones(3).astype(np.float32)
# z.backward()
# print(x.grad)

''' 3.2.3 '''
h = L.Linear(3, 4)
x = Variable(np.array(range(6)).astype(np.float32).reshape(2, 3))
print(h.W.data)
print(x.data)
print()
print(h.b.data)
print()
y = h(x)
print(y.data)

w = h.W.data
x0 = x.data
print(x0.dot(w.T) + h.b.data)
