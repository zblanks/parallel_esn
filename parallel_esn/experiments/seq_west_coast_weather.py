import argparse
import numpy as np
import matplotlib.pyplot as plt
import time
from ..esn import ESN
from ..utils import to_forecast_form, standardize_traindata, scale_data, unscale_data
from ..bo import BO

"""
Attempts to predict temperature, humidity, and pressure for 5 west coast cities,
Vancouver, Seattle, Portland, San Francisco, and Los Angeles.

Sequential version of the code.
"""


def prep_data(filename, in_len, pred_len):
    """load data from the file and split it into windows of input"""
    data = np.genfromtxt(filename, delimiter=',', skip_header=1,
                         usecols=np.arange(1, 16), dtype=float)

    # Remove rows that are missing values (none should be, generally)
    origlen = data.shape[0]
    data = data[~np.isnan(data).any(axis=1)]
    newlen = data.shape[0]
    print("Discarded {} datapoints out of {}".format(origlen-newlen, origlen))

    # We will save the last 1/8th of the data for validation/testing data,
    # 1/16 for validation, 1/16 for testing
    total_len = data.shape[0]
    val_len = total_len // 16
    test_len = total_len // 16
    train_len = total_len - val_len - test_len

    train_data = data[:train_len]
    val_data = data[train_len:train_len + val_len]
    test_data = data[train_len + val_len:]

    # To stay in the most accurate ranges of the ESN, and to put the various
    # features on equal footing, we standardize the training data.
    train_data, mu_arr, sigma_arr = standardize_traindata(train_data)

    # We now need to scale our validation and test data by the means and standard
    # deviations determined from the training data
    val_data = scale_data(val_data, mu_arr, sigma_arr)
    test_data = scale_data(test_data, mu_arr, sigma_arr)

    # We need to convert the time series data to forecast form for one-step
    # prediction training. For simplicity we will discard the remainder batches
    # which are returned by to_forecast_from since we won't be losing much data
    # anyways.
    train_batch_size = 200
    val_batch_size = in_len + pred_len + 1
    test_batch_size = in_len + pred_len + 1
    trainU, trainY, _, _ = to_forecast_form(train_data, batch_size=train_batch_size)
    valU, valY, _, _ = to_forecast_form(val_data, batch_size=val_batch_size, stride=pred_len)
    testU, testY, _, _ = to_forecast_form(test_data, batch_size=test_batch_size, stride=pred_len)

    return trainU, trainY, valU, valY, testU, testY, mu_arr, sigma_arr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_iter', type=int, nargs='?', default=1)
    parser.add_argument('--filename', type=str, nargs='?', default='west_coast_weather.csv')
    parser.add_argument('--verbose', type=int, nargs='?', default=0)
    args = parser.parse_args()

    # Get the start time for the run
    start_time = time.time()

    in_len = 100
    pred_len = 24

    trainU, trainY, valU, valY, testU, testY, mu, sigma = prep_data(args.filename,
                                                                    in_len, pred_len)

    bo = BO(k=(2, 50), hidden_dim=(400, 420), random_state=12)
    # for reproducibility
    np.random.seed(12)

    best_loss = 1e8
    best_esn = None

    time_arr = np.arange(in_len+pred_len+1)

    # To choose which runs to look at at random
    if testU.shape[0] >= 9:
        replace = False
    else:
        replace = True
    wi = np.random.choice(testU.shape[0], 9, replace=replace)

    for i in range(args.num_iter):
        h_star = bo.find_best_choices()
        print("Iteration {}".format(i))
        print(h_star)

        esn = ESN(input_dim=trainU.shape[1], hidden_dim=h_star['hidden_dim'],
                  output_dim=trainY.shape[1], k=h_star['k'],
                  spectral_radius=h_star['spectral_radius'],
                  p=h_star['p'], alpha=h_star['alpha'], beta=h_star['beta'])

        val_loss = esn.recursive_train_validate(trainU, trainY, valU, valY,
                                                in_len, pred_len, verbose=args.verbose,
                                                compute_loss_freq=10)
        print("validation loss = {}".format(val_loss))

        if val_loss < best_loss:
            best_esn = esn
            best_loss = val_loss

        bo.update_gpr(X=[h_star[val] for val in h_star.keys()], y=val_loss)

    # Get the end time for the run
    end_time = time.time()
    elapsed_time = end_time - start_time

    print("Sequential: Elapsed time = {} sec".format(elapsed_time))

    # Humidity Figure
    fig_h, ax_h = plt.subplots(3, 3, figsize=(15, 14))
    ax_h = ax_h.flatten()

    # Temperature Figure
    fig_t, ax_t = plt.subplots(3, 3, figsize=(15, 14))
    ax_t = ax_t.flatten()

    # Pressure Figure
    fig_p, ax_p = plt.subplots(3, 3, figsize=(15, 14))
    ax_p = ax_p.flatten()

    for k in range(len(wi)):
        dat_in = unscale_data(testU[wi[k], :, :in_len].T, mu, sigma)
        ax_t[k].plot(time_arr[:in_len], dat_in[:, 3], 'ob', label='input')
        ax_h[k].plot(time_arr[:in_len], dat_in[:, 8], 'ob', label='input')
        ax_p[k].plot(time_arr[:in_len], dat_in[:, 13], 'ob', label='input')
    for k in range(len(wi)):
        dat_obs = unscale_data(testU[wi[k], :, in_len:in_len+pred_len].T, mu, sigma)
        ax_t[k].plot(time_arr[in_len:in_len+pred_len], dat_obs[:, 3], '^g', label='observed')
        ax_h[k].plot(time_arr[in_len:in_len+pred_len], dat_obs[:, 8], '^g', label='observed')
        ax_p[k].plot(time_arr[in_len:in_len+pred_len], dat_obs[:, 13], '^g', label='observed')
        s_pred = best_esn.recursive_predict(testU[wi[k], :, :in_len], pred_len)
        dat_pred = unscale_data(s_pred.T, mu, sigma)
        ax_t[k].plot(time_arr[in_len:in_len+pred_len], dat_pred[:, 3], '-r', label="Best ESN")
        ax_h[k].plot(time_arr[in_len:in_len+pred_len], dat_pred[:, 8], '-r', label="Best ESN")
        ax_p[k].plot(time_arr[in_len:in_len+pred_len], dat_pred[:, 13], '-r', label="Best ESN")
    plt.figure(fig_t.number)
    plt.suptitle("SF Temperature")
    plt.figure(fig_h.number)
    plt.suptitle("SF Humidity")
    plt.figure(fig_p.number)
    plt.suptitle("SF Pressure")
    for i in range(len(ax_h)):
        ax_t[i].set_xlim(time_arr[in_len - 2*pred_len], time_arr[in_len + pred_len])
        ax_h[i].set_xlim(time_arr[in_len - 2*pred_len], time_arr[in_len + pred_len])
        ax_p[i].set_xlim(time_arr[in_len - 2*pred_len], time_arr[in_len + pred_len])
    for idx in [0, 3, 6]:
        ax_t[idx].set_ylabel("Temperature (Kelvin)")
        ax_h[idx].set_ylabel("Humidity (percent)")
        ax_p[idx].set_ylabel("Pressure (mb)")
    for idx in [6, 7, 8]:
        ax_t[idx].set_xlabel("Hours")
        ax_h[idx].set_xlabel("Hours")
        ax_p[idx].set_xlabel("Hours")
    ax_h[0].legend(loc=2, numpoints=1)
    plt.show()


if __name__ == '__main__':
    main()
