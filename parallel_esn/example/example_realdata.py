from pkg_resources import resource_filename
import numpy as np
import matplotlib.pyplot as plt
from ..esn import ESN
from ..utils import chunk_data, standardize_traindata, scale_data

# Load data
fname = resource_filename('parallel_esn', 'data/PJM_Load_hourly.csv')
data = np.loadtxt(fname, delimiter=',', skiprows=1, usecols=[1])

tot_len = data.shape[0]
val_len = tot_len//10

train_len = tot_len-val_len

# Split up loaded data with 9/10ths going to training data
# and 1/10th going to validation data
train_dat = data[:train_len]
val_dat = data[train_len:]

# Standardize training data to make it more neural network-friendly
train_dat, mu, sigma = standardize_traindata(train_dat)
# Scale validatino data by mean and s.dev determined by training data
val_dat = scale_data(val_dat, mu, sigma)

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
plt.title("PJM Standardized Power Consumption")
plt.ylabel("Arb. Units.")
plt.xlabel("Hours")
plt.legend(loc=2, numpoints=1)
plt.show()
