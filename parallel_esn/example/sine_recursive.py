import numpy as np
from numpy.random import normal
import matplotlib.pyplot as plt
from ..esn import ESN
from ..utils import to_forecast_form

# Create a noisy sinusoid:
np.random.seed(17)  # Set seed for deterministic results
sigma = 0.3
t = np.linspace(0, 1000, 10001)
size = len(t)
data = np.sin(2*np.pi*t) + normal(0., sigma, size)

# Create a validation the same way, with a phase shift
val_t = np.linspace(0, 300, 3001)
val_size = len(val_t)
val_data = np.sin(2*np.pi*t + np.sqrt(2)) + normal(0., sigma, size)

trainU, trainY, remainderU, remainderY = to_forecast_form(data, -1)
valU, valY, remainderU, remainderY = to_forecast_form(val_data, -1)

# Create a new ESN
esn = ESN(1, 60, 1, 3)
loss = esn.train_validate(trainU, trainY, valU, valY)
print("validation loss = {}".format(loss))

num_pred = 100
plt.plot(t[:30], valU[0, 0, :30], 'ob', label='input')
pred = esn.recursive_predict(valU[0, 0:1, :30], num_pred, cold_start=True)
plt.plot(t[30:30+num_pred], pred[0, :], '-r', label='predicted')
plt.plot(t[30:30+num_pred], valU[0, 0, 30:30+num_pred], '^g', label='observed')
plt.legend(loc=2, numpoints=1)
plt.show()
