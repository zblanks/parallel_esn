import numpy as np
import matplotlib.pyplot as plt
from ..esn import ESN
from ..utils import chunk_data

# Load data

data = np.loadtxt('PJM_Load_hourly.csv', delimiter=',', skiprows=1, usecols=[1])

tot_len = data.shape[0]
val_len = tot_len//10

train_len = tot_len-val_len

data = data - np.average(data)
data /= np.std(data)

train_dat = data[:train_len]
val_dat = data[train_len:]

windowsize = 160
trainU, trainY = chunk_data(train_dat, windowsize, 20)
valU, valY = chunk_data(val_dat, windowsize, 20)

# Create a new ESN
esn = ESN(1, windowsize, 1, 3)
loss = esn.train_validate(trainU, trainY, valU, valY)
print("validation loss = {}".format(loss))

time = np.arange(windowsize)
plt.plot(time, valU[0, 0, :], 'ob', label='input')
pred = esn.predict(valU[0, 0:1, :])
plt.plot(time+windowsize, pred[0, :], '-r', label='predicted')
plt.plot(time+windowsize, valY[0, 0, :], '^g', label='observed')
plt.title("PJM Normalized Power Consumption")
plt.ylabel("Arb. Units.")
plt.xlabel("Hours")
plt.legend(loc=2, numpoints=1)
plt.show()
