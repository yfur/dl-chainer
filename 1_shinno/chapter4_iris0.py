import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, Variable
from chainer import optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from sklearn import datasets
import time

''' 1. Data preparation and settings '''
# Load iris dataset from scikit-learn
iris = datasets.load_iris()
X = iris.data.astype(np.float32)
Y = iris.target
N = Y.size
Y2 = np.zeros(3 * N).reshape(N,3).astype(np.float32)
for i in range(N):
    Y2[i,Y[i]] = 1.0

index = np.arange(N)
# train data with odd index
xtrain = X[index[index % 2 != 0], :]
ytrain = Y2[index[index % 2 != 0], :]
# test data with even index
xtest = X[index[index % 2 == 0],:]
yans = Y[index[index % 2 == 0]]

''' 2. Definition of a model with Chain class '''
class IrisChain(Chain):
    def __init__(self):
        super(IrisChain, self).__init__(
            l1=L.Linear(4,6),
            l2=L.Linear(6,3),
        )

    def __call__(self,x,y):
        return F.mean_squared_error(self.fwd(x), y)

    def fwd(self,x):
         h1 = F.sigmoid(self.l1(x))
         h2 = self.l2(h1)
         return h2


''' 3. Initialization '''
model = IrisChain()
optimizer = optimizers.SGD()
optimizer.setup(model)

''' 4. Learning '''
for i in range(10000):
    x = Variable(xtrain)
    y = Variable(ytrain)
    model.zerograds()
    loss = model(x,y)
    loss.backward()
    optimizer.update()

''' 5. Testing '''
xt = Variable(xtest, volatile='on')
yy = model.fwd(xt)

ans = yy.data
nrow, ncol = ans.shape
ok = 0
for i in range(nrow):
    cls = np.argmax(ans[i,:])
    print(ans[i,:], cls)
    if cls == yans[i]:
        ok += 1


print('{0:d} / {1:d} = {2:f}'.format(ok, nrow, ok/nrow))
