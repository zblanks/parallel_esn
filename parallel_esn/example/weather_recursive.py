import numpy as np
import argparse
import matplotlib.pyplot as plt
from ..esn import ESN
from ..utils import to_forecast_form, standardize_traindata, scale_data, unscale_data
from ..bo import BO

"""
Attempts to predict a window of humidities in a recursive manner, producing more
accurate results for near term.
"""
def prep_data(filename, in_len, pred_len):
    """load data from the file and chunk it into windows of input"""
    # Columns are
    # 0:datetime, 1:temperature, 2:humidity, 3:pressure, 4:wind_direction, 5:wind_speed
    data = np.genfromtxt(filename, delimiter=',', skip_header=1,
                         usecols=(1, 2, 3, 4, 5), dtype=float)

    # Remove rows that are missing values
    data = data[~np.isnan(data).any(axis=1)]

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
    # prediction training. For simplicity we will discard the remainder batches rU, rY
    train_batch_size = 200
    val_batch_size = 200
    test_batch_size = in_len + pred_len + 1
    trainU, trainY, rU, rY = to_forecast_form(train_data, batch_size=train_batch_size)
    valU, valY, rU, rY = to_forecast_form(val_data, batch_size=val_batch_size)
    testU, testY, rU, rY = to_forecast_form(test_data, batch_size=test_batch_size)

    return trainU, trainY, valU, valY, testU, testY, mu_arr, sigma_arr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_len', type=int, nargs='?', default=200)
    parser.add_argument('--pred_len', type=int, nargs='?', default=24)
    parser.add_argument('--num_iter', type=int, nargs='?', default=5)
    parser.add_argument('--filename', type=str, nargs='?', default='boston_weather.csv')
    args = parser.parse_args()

    in_len = args.input_len
    pred_len = args.pred_len

    trainU, trainY, valU, valY, testU, testY, mu, sigma = prep_data(args.filename,
                                                                    in_len, pred_len)

    bo = BO(k=(2, 50), hidden_dim=(200, 400), random_state=12)
    # for reproducibility
    np.random.seed(12)

    best_loss = 1e8
    best_esn = None
    
    time = np.arange(in_len+pred_len+1)

    # To choose which runs to look at at random
    if testU.shape[0] >= 9:
        replace = False
    else:
        replace = True
    wi = np.random.choice(testU.shape[0], 9, replace=replace)

    # Humidity Figure
    fig_h, ax_h = plt.subplots(3, 3, figsize=(15,14))
    ax_h = ax_h.flatten()

    # Temperature Figure
    fig_t, ax_t = plt.subplots(3, 3, figsize=(15,14))
    ax_t = ax_t.flatten()
    for k in range(len(wi)):
        dat_in = unscale_data(testU[wi[k], :, :in_len].T, mu, sigma)
        ax_t[k].plot(time[:in_len], dat_in[:, 0], 'ob', label='input')
        ax_h[k].plot(time[:in_len], dat_in[:, 1], 'ob', label='input')
    for i in range(args.num_iter):
        H_space = bo.build_options()
        h_star = bo.find_best_choice(H_space)
        print("Iteration {}".format(i))
        print(h_star)

        esn = ESN(input_dim=trainU.shape[1], hidden_dim=h_star['hidden_dim'],
                  output_dim=trainY.shape[1], k=h_star['k'],
                  spectral_radius=h_star['spectral_radius'],
                  p=h_star['p'], alpha=h_star['alpha'], beta=h_star['beta'])

        val_loss = esn.train_validate(trainU, trainY, valU, valY, verbose=1, compute_loss_freq=10)
        print("validation loss = {}".format(val_loss))

        for k in range(len(wi)):
            s_pred = esn.recursive_predict(testU[wi[k], :, :in_len], pred_len)
            dat_pred = unscale_data(s_pred.T, mu, sigma)
            ax_t[k].plot(time[in_len:in_len+pred_len], dat_pred[:, 0], '-', color='#888888', alpha=0.1) 
            ax_h[k].plot(time[in_len:in_len+pred_len], dat_pred[:, 1], '-', color='#888888', alpha=0.1) 
        if val_loss < best_loss:
            best_esn = esn
            best_loss = val_loss

        bo.update_gpr(X=[h_star[val] for val in h_star.keys()], y=val_loss)
    for k in range(len(wi)):
        dat_obs = unscale_data(testU[wi[k], :, in_len:in_len+pred_len].T, mu, sigma)
        ax_t[k].plot(time[in_len:in_len+pred_len], dat_obs[:, 0], '^g', label='observed')
        ax_h[k].plot(time[in_len:in_len+pred_len], dat_obs[:, 1], '^g', label='observed')
        s_pred = best_esn.recursive_predict(testU[wi[k], :, :in_len], pred_len)
        dat_pred = unscale_data(s_pred.T, mu, sigma)
        ax_t[k].plot(time[in_len:in_len+pred_len], dat_pred[:, 0], '-r', label="Best ESN") 
        ax_h[k].plot(time[in_len:in_len+pred_len], dat_pred[:, 1], '-r', label="Best ESN") 
    plt.figure(fig_t.number)
    plt.suptitle("Boston Temperature, Recursive Prediction")
    plt.figure(fig_h.number)
    plt.suptitle("Boston Humidity, Recursive Prediction")
    for i in range(len(ax_h)):
        ax_t[i].set_xlim(time[in_len - 2*pred_len], time[in_len + pred_len])
        ax_h[i].set_xlim(time[in_len - 2*pred_len], time[in_len + pred_len])
    for idx in [0, 3, 6]:
        ax_t[idx].set_ylabel("Temperature (Kelvin)")
        ax_h[idx].set_ylabel("Humidity (percent)")
    for idx in [6, 7, 8]:
        ax_t[idx].set_xlabel("Hours")
        ax_h[idx].set_xlabel("Hours")
    ax_h[0].legend(loc=2, numpoints=1)
    plt.show()


if __name__ == '__main__':
    main()
