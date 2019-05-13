from pkg_resources import resource_filename
import numpy as np
import time
import matplotlib.pyplot as plt
from ..esn import ESN
from ..utils import to_forecast_form, standardize_traindata, scale_data

# An example using recursive forecasting

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
# Scale validation data by mean and s.dev determined by training data
val_dat = scale_data(val_dat, mu, sigma)

batch_size = 400
# Convert data to forecast form
trainU, trainY, rU, rY = to_forecast_form(train_dat, batch_size)
# Batch size of -1 will place all data in a single batch.
valU, valY, rU, rY = to_forecast_form(val_dat, -1)

# Create a new ESN
start_time = time.time()
esn = ESN(1, 2400, 1, k=20, alpha=0.8, use_cython=False, use_sparse=True)
loss = esn.train_validate(trainU, trainY, valU, valY)
print("validation loss = {}".format(loss))
end_time = time.time()
print("Time taken: {} sec".format(end_time - start_time))

time_arr = np.arange(valU.shape[2])
input_len = 180
pred_len = 24
plt.plot(time_arr[:input_len], valU[0, 0, :input_len], 'ob', label='input')
pred = esn.recursive_predict(valU[0, 0:1, :input_len], pred_len)
plt.plot(time_arr[input_len:input_len+pred_len], pred[0, :], '-r', label='predicted')
plt.plot(time_arr[input_len:input_len+pred_len], valU[0, 0, input_len:input_len+pred_len], '^g', label='observed')
plt.xlim(time_arr[input_len+pred_len - 4*pred_len], time_arr[input_len+pred_len])
plt.title("PJM Standardized Power Consumption (Recursive 1 Step Forecast)")
plt.ylabel("Arb. Units.")
plt.xlabel("Hours")
plt.legend(loc=2, numpoints=1)
plt.show()
