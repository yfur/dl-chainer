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
Y = iris.target.astype(np.int32)
N = Y.size

index = np.arange(N)
# train data with odd index
xtrain = X[index[index % 2 != 0], :]
ytrain = Y[index[index % 2 != 0]]
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
        return F.softmax_cross_entropy(self.fwd(x), y) # softmax cross entropy

    def fwd(self,x):
         h1 = F.sigmoid(self.l1(x))
         h2 = self.l2(h1)
         return h2


''' 3. Initialization '''
model = IrisChain()
optimizer = optimizers.SGD()
optimizer.setup(model)

''' 4. Learning '''
# batch
n = 75
bs = 25
for j in range(5000):
    accum_loss = None
    sffindex = np.random.permutation(n)
    for i in range(0, n, bs):
        x = Variable(xtrain[sffindex[i:(i + bs) if (i + bs) < n else n]])
        y = Variable(ytrain[sffindex[i:(i + bs) if (i + bs) < n else n]])
        model.zerograds()
        loss = model(x,y)
        accum_loss = loss if accum_loss is None else accum_loss + loss
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
print(ans[0])
