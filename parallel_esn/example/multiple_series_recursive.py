import numpy as np
from numpy.random import normal
import argparse
import matplotlib.pyplot as plt
from ..esn import ESN
from ..utils import to_forecast_form, standardize_traindata, scale_data, unscale_data

"""
The goal of this example is to show that feeding in the time series multiple features
to the echo state network can help in predictions.

Sample data that we would like to predict is generated as a sum of three sine
waves and gaussian noise.

In the baseline example, the ESN only has access to this single time series
In the improved version, the ESN gets the aforementioned time series and
a second time series which is correlated with the first (a noise-free sine
wave of the same frequency, but different phase, as one of the sine waves
present in the first time series).

The validation loss is decreased when the ESN has both time series to work
with, as would be expected in the case where more information is available.

"""


def prep_data():
    """Generate data from the file and chunk it into windows of input"""
    time = np.arange(1000, dtype=float)

    # First time series is made up of sines and noise
    data1 = np.sin(np.pi / 24. * time) + 0.6 * np.sin(np.pi / 7.9 * time) \
                                       + 0.3 * np.sin(np.pi / 2.9 * time)
    data1 += normal(0., 0.01, len(data1))

    # Second time series is one of the sines present in the first, but
    # with a phase offset.
    data2 = np.sin(np.pi / 2.9 * time + np.sqrt(2))
    data = np.stack((data1, data2), axis=-1)

    # We will save the last 1/4th of the data for validation/testing data,
    total_len = data.shape[0]
    val_len = total_len // 4
    train_len = total_len - val_len

    train_data = data[:train_len]
    val_data = data[train_len:]

    # To stay in the most accurate ranges of the ESN, and to put the various
    # features on equal footing, we standardize the training data.
    train_data, mu_arr, sigma_arr = standardize_traindata(train_data)

    # We now need to scale our validation data by the means and standard
    # deviations determined from the training data
    val_data = scale_data(val_data, mu_arr, sigma_arr)

    # Convert the time series data to forecast form for one-step prediction training
    trainU, trainY, rU, rY = to_forecast_form(train_data)
    valU, valY, rU, rY = to_forecast_form(val_data)
    # We won't worry about the remainder data in rU and rY since they are empty

    return trainU, trainY, valU, valY, mu_arr, sigma_arr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_len', type=int, nargs='?', default=72)
    parser.add_argument('--pred_len', type=int, nargs='?', default=12)
    args = parser.parse_args()

    trainU, trainY, valU, valY, mu, sigma = prep_data()

    # Which batch of validation data to plot
    wi = 0
    input_len = args.input_len
    pred_len = args.pred_len
    time = np.arange(input_len+pred_len+1)
    temp_in = unscale_data(valU[wi, 0:1, :input_len].T, mu, sigma)
    temp_obs = unscale_data(valU[wi, 0:1, input_len:input_len+pred_len].T,
                            mu, sigma)

    plt.figure(figsize=(7, 5))
    plt.plot(time[:input_len], temp_in[:, 0], 'ob', label='input')
    plt.plot(time[input_len:input_len+pred_len], temp_obs[:, 0], '^g', label='observed')

    # Baseline ESN that only gets to train on one timeseries
    esn_baseline = ESN(input_dim=1, hidden_dim=200, output_dim=1, k=3)

    # ESN that only gets to train on both timeseries
    esn = ESN(input_dim=trainU.shape[1], hidden_dim=200, output_dim=trainY.shape[1], k=3)

    val_loss_baseline = esn_baseline.train_validate(trainU[:, 0:1, :], trainY[:, 0:1, :],
                                                    valU[:, 0:1, :], valY[:, 0:1, :], verbose=1)
    val_loss = esn.train_validate(trainU, trainY, valU, valY, verbose=1)
    print("baseline validation loss = {}".format(val_loss_baseline))
    print("validation loss = {}".format(val_loss))

    s_pred_baseline = esn_baseline.recursive_predict(valU[wi, 0:1, :input_len], pred_len)
    temp_pred_baseline = unscale_data(s_pred_baseline.T, mu, sigma)
    plt.plot(time[input_len:input_len+pred_len], temp_pred_baseline[:, 0], '-y',
             label=("predicted (single timeseries)\nloss = {0:.3f}".format(val_loss_baseline)))

    s_pred = esn.recursive_predict(valU[wi, :, :input_len], pred_len)
    temp_pred = unscale_data(s_pred.T, mu, sigma, predict_cols=[0])
    plt.plot(time[input_len:input_len+pred_len], temp_pred[:, 0], '-r',
             label=("predicted (double timeseries)\n"
                    "loss = {0:.3f}".format(val_loss)))

    plt.title("Prediction power gain using correlated timeseries (recursive)")
    plt.xlim(time[input_len - 2*pred_len], time[input_len + pred_len])
    plt.legend(bbox_to_anchor=(0.8, 0), loc='lower left', numpoints=1)
    plt.subplots_adjust(right=0.7)
    plt.show()


if __name__ == '__main__':
    main()
