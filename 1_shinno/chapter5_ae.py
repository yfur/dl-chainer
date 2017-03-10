import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, Variable
from chainer import optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from sklearn import datasets
import matplotlib.pyplot as plt

''' 1. Data preparation and settings '''
# Load iris dataset from scikit-learn
iris = datasets.load_iris()
xtrain = iris.data.astype(np.float32)

''' 2. Definition of a model with Chain class '''
# Auto Encoder
class MyAE(Chain):
    def __init__(self):
        super(MyAE, self).__init__(
            l1=L.Linear(4,2),
            l2=L.Linear(2,4),
        )

    def __call__(self,x):
        bv = self.fwd(x)
        return F.mean_squared_error(bv, x)

    def fwd(self,x):
        fv = F.sigmoid(self.l1(x))
        bv = self.l2(fv)
        return bv

''' 3. Initialization '''
model = MyAE()
optimizer = optimizers.SGD()
optimizer.setup(model)

''' 4. Learning '''
n = 150
bs = 30
for j in range(3000):
    sffindx = np.random.permutation(range(n))
    for i in range(0, n, bs):
        x = Variable(xtrain[sffindx[i:(i+bs) if (i+bs) < n else n]])
        model.zerograds()
        loss = model(x)
        loss.backward()
        optimizer.update()

''' 5. Testing '''
x = Variable(xtrain, volatile='on')
yt = F.sigmoid(model.l1(x))
ans = yt.data
for i in range(n):
    print(ans[i,:])

x = Variable(xtrain)
yt = F.sigmoid(model.l1(x))
ans = yt.data
ansx1 = ans[0:50, 0]
ansy1 = ans[0:50, 1]
ansx2 = ans[50:100, 0]
ansy2 = ans[50:100, 1]
ansx3 = ans[100:150, 0]
ansy3 = ans[100:150, 1]
plt.close()
plt.figure()
plt.scatter(ansx1, ansy1, marker='^')
plt.scatter(ansx2, ansy2, marker='o')
plt.scatter(ansx3, ansy3, marker='+')
plt.savefig('./1_shinno/chapter5_output/ae.pdf', format='pdf')
plt.show()
