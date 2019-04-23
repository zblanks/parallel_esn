import numpy as np
from numpy.random import normal
import matplotlib.pyplot as plt
from ..esn import ESN
from ..utils import chunk_data

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

trainU, trainY = chunk_data(data, 30, 30)
valU, valY = chunk_data(val_data, 30, 30)

# Create a new ESN
esn = ESN(1, 60, 1, 3)
loss = esn.train_validate(trainU, trainY, valU, valY)
print("validation loss = {}".format(loss))

plt.plot(t[:30], valU[0, 0, :], 'ob', label='input')
pred = esn.predict(valU[0, 0:1, :])
plt.plot(t[30:60], pred[0, :], '-r', label='predicted')
plt.plot(t[30:60], valY[0, 0, :], '^g', label='observed')
plt.legend(loc=2, numpoints=1)
plt.show()
