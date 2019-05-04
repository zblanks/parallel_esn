"""
Attempts to predict temperature, humidity, and pressure for 5 west coast cities,
Vancouver, Seattle, Portland, San Francisco, and Los Angeles.

Sequential version of the code.
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from ..esn import ESN
from ..utils import to_forecast_form, standardize_traindata, scale_data, unscale_data
from ..bo import BO


def prep_data(filename, in_len, pred_len):
    """load data from the file and split it into windows of input"""
    data = np.genfromtxt(filename, delimiter=',', skip_header=1,
                         usecols=np.arange(1, 16), dtype=float)

    # Remove rows that are missing values (none should be, generally)
    origlen = data.shape[0]
    data = data[~np.isnan(data).any(axis=1)]
    newlen = data.shape[0]
    print("Discarded {} datapoints out of {}".format(origlen-newlen, origlen))

    # We will save the last 1/10th of the data for validation/testing data,
    # 1/8th of that for testing, rest for validation
    total_len = data.shape[0]
    val_test_len = total_len // 10
    test_len = val_test_len // 8
    val_len = val_test_len - test_len
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
    stride = pred_len//2
    trainU, trainY, _, _ = to_forecast_form(train_data, batch_size=train_batch_size)
    valU, valY, _, _ = to_forecast_form(val_data, batch_size=val_batch_size, stride=stride)
    testU, testY, _, _ = to_forecast_form(test_data, batch_size=test_batch_size, stride=stride)

    return trainU, trainY, valU, valY, testU, testY, mu_arr, sigma_arr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_iter', type=int, nargs='?', default=1)
    parser.add_argument('--filename', type=str, nargs='?', default='west_coast_weather.csv')
    parser.add_argument('--outdir', type=str, nargs='?', default='figs')
    parser.add_argument('--verbose', type=int, nargs='?', default=0)
    args = parser.parse_args()

    # Get the start time for the run
    start_time = time.time()

    in_len = 100
    pred_len = 24

    trainU, trainY, valU, valY, testU, testY, mu, sigma = prep_data(args.filename,
                                                                    in_len, pred_len)

    bo = BO(k=(2, 100), hidden_dim=(400, 800), random_state=12)
    # for reproducibility
    np.random.seed(12)

    best_loss = 1e8
    best_esn = None

    time_arr = np.arange(in_len+pred_len+1)

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

    print("Validation set size: {}".format(valU.shape[0]))
    print("Test set size: {}".format(testU.shape[0]))

    # Compute test error
    test_loss = esn.recursive_validate(testU, valU, in_len, pred_len, verbose=args.verbose)

    print("test loss = {}".format(test_loss))

    # Make figures directory, ok if it already exists
    os.makedirs(args.outdir, exist_ok=True)

    # To choose which runs to plot at random
    if testU.shape[0] >= 9:
        replace = False
    else:
        replace = True
    wi = np.random.choice(testU.shape[0], 9, replace=replace)

    city = ['Vancouver', 'Portland', 'San Francisco', 'Seattle', 'Los Angeles']
    for ii in range(5):
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
            ax_t[k].plot(time_arr[:in_len], dat_in[:, ii], 'ob', label='input')
            ax_h[k].plot(time_arr[:in_len], dat_in[:, ii+5], 'ob', label='input')
            ax_p[k].plot(time_arr[:in_len], dat_in[:, ii+10], 'ob', label='input')
        for k in range(len(wi)):
            dat_obs = unscale_data(testU[wi[k], :, in_len:in_len+pred_len].T, mu, sigma)
            ax_t[k].plot(time_arr[in_len:in_len+pred_len], dat_obs[:, ii], '^g', label='observed')
            ax_h[k].plot(time_arr[in_len:in_len+pred_len], dat_obs[:, ii+5], '^g', label='observed')
            ax_p[k].plot(time_arr[in_len:in_len+pred_len], dat_obs[:, ii+10], '^g', label='observed')
            s_pred = best_esn.recursive_predict(testU[wi[k], :, :in_len], pred_len)
            dat_pred = unscale_data(s_pred.T, mu, sigma)
            ax_t[k].plot(time_arr[in_len:in_len+pred_len], dat_pred[:, ii], '-r', label="Best ESN")
            ax_h[k].plot(time_arr[in_len:in_len+pred_len], dat_pred[:, ii+5], '-r', label="Best ESN")
            ax_p[k].plot(time_arr[in_len:in_len+pred_len], dat_pred[:, ii+10], '-r', label="Best ESN")
        plt.figure(fig_t.number)
        plt.suptitle(city[ii] + " Temperature", y=0.9, fontsize=14)
        plt.figure(fig_h.number)
        plt.suptitle(city[ii] + " Humidity", y=0.9, fontsize=14)
        plt.figure(fig_p.number)
        plt.suptitle(city[ii] + " Pressure", y=0.9, fontsize=14)
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
        ax_t[0].legend(loc=2, numpoints=1)
        ax_h[0].legend(loc=2, numpoints=1)
        ax_p[0].legend(loc=2, numpoints=1)

        cityfile = args.outdir.rstrip('/') + '/' + city[ii].replace(' ', '')
        print("Saving: " + cityfile)
        plt.figure(fig_t.number)
        plt.savefig(cityfile + "_temp.pdf", bbox_inches='tight')
        plt.figure(fig_h.number)
        plt.savefig(cityfile + "_humid.pdf", bbox_inches='tight')
        plt.figure(fig_p.number)
        plt.savefig(cityfile + "_press.pdf", bbox_inches='tight')

        plt.close('all')


if __name__ == '__main__':
    main()
