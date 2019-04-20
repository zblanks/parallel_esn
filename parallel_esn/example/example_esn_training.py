import numpy as np
from numpy.random import normal
from ..esn import ESN
import matplotlib.pyplot as plt

# Create a noisy sinusoid:

sigma = 0.3
t = np.linspace(0,1000,10000)
size = len(t)
data = np.sin(2*np.pi*t) + normal(0.,sigma,size)

# Create a validation the same way, with a phase shift
val_t = np.linspace(0,300,0.1)
val_size = len(val_t)
val_data = np.sin(2*np.pi*t + np.sqrt(2)) + normal(0.,sigma,size)

def chunk_data(timeseries, windowsize):
    length = timeseries.shape[0]
    num_chunks = length//(2*windowsize)
    batchU = np.zeros((num_chunks,1,windowsize))
    batchY = np.zeros((num_chunks,1,windowsize))
    for i in range(num_chunks):
        start = 2*windowsize*i
        end = start + windowsize
        batchU[i,0,:] = timeseries[start:end]
        start = 2*windowsize*i + windowsize
        end = start + windowsize
        batchY[i,0,:] = timeseries[start:end]
    return batchU, batchY

trainU, trainY = chunk_data(data,30)
valU, valY = chunk_data(val_data,30)


# Create a new ESN
esn = ESN(1, 60, 1,3)
loss = esn.train_validate(trainU, trainY, valU, valY)
print("validation loss = {}".format(loss))

plt.plot(t[:30],valU[0,0,:],'ob',label='input')
pred = esn.predict(valU[0,0:1,:])
plt.plot(t[30:60],pred[0,:],'-r',label='predicted')
plt.plot(t[30:60],valY[0,0,:],'^g',label='observed')
plt.legend(loc=2,numpoints=1)
plt.show()
