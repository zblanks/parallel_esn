import numpy as np
import argparse
import matplotlib.pyplot as plt
from ..esn import ESN
from ..utils import chunk_data, standardize_traindata, scale_data, unscale_data
from ..bo import BO

"""
Attempts to predict a window of humidities in a one shot (multiple output) manner
"""


def prep_data(filename, windowsize):
    """load data from the file and chunk it into windows of input"""
    # Columns are
    # 0:datetime, 1:temperature, 2:humidity, 3:pressure, 4:wind_direction, 5:wind_speed
    data = np.genfromtxt(filename, delimiter=',', skip_header=1,
                         usecols=(1, 2, 3, 4, 5), dtype=float)

    # Remove rows that are missing values
    data = data[~np.isnan(data).any(axis=1)]

    # We will save the last 1/8th of the data for validation/testing data,
    # 3/32 for validation, 1/32 for testing
    total_len = data.shape[0]
    val_len = 3 * (total_len // 32)
    test_len = total_len // 32
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

    # We chunk the training data, but we only want to predict the humidity
    # which is in column 1 of data
    trainU, trainY = chunk_data(train_data, windowsize, predict_cols=[1])
    valU, valY = chunk_data(val_data, windowsize, predict_cols=[1])
    testU, testY = chunk_data(test_data, windowsize, predict_cols=[1])

    return trainU, trainY, valU, valY, testU, testY, mu_arr, sigma_arr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--windowsize', type=int, nargs='?', default=24)
    parser.add_argument('--num_iter', type=int, nargs='?', default=5)
    parser.add_argument('--filename', type=str, nargs='?', default='boston_weather.csv')
    args = parser.parse_args()

    trainU, trainY, valU, valY, testU, testY, mu, sigma = prep_data(args.filename,
                                                                    args.windowsize)

    bo = BO(k=(2, 50), hidden_dim=(200, 400), random_state=12)

    best_loss = 1e8
    best_esn = None

    time = np.arange(args.windowsize)
    # wi = [0,9,16,39,53,79,101,122,134]
    wi = np.random.choice(testU.shape[0], 9, replace=False)
    fig, ax = plt.subplots(3, 3, figsize=(15, 14))
    ax = ax.flatten()
    for k in range(len(wi)):
        hum_in = unscale_data(testU[wi[k], 1:2, :].T, mu, sigma, predict_cols=[1])
        ax[k].plot(time, hum_in[:, 0], 'ob', label='input')
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
            s_pred = esn.predict(testU[wi[k], :, :])
            hum_pred = unscale_data(s_pred.T, mu, sigma, predict_cols=[1])
            ax[k].plot(time+args.windowsize, hum_pred[:, 0], '-', color='#BBBBBB')
        if val_loss < best_loss:
            best_esn = esn
            best_loss = val_loss

        bo.update_gpr(X=[h_star[val] for val in h_star.keys()], y=val_loss)
    for k in range(len(wi)):
        hum_obs = unscale_data(testY[wi[k], 0:1, :].T, mu, sigma, predict_cols=[1])
        ax[k].plot(time+args.windowsize, hum_obs[:, 0], '^g', label='observed')
        s_pred = best_esn.predict(testU[wi[k], :, :])
        hum_pred = unscale_data(s_pred.T, mu, sigma, predict_cols=[1])
        ax[k].plot(time+args.windowsize, hum_pred[:, 0], '-r', label="Best ESN")
    plt.suptitle("Boston Humidity, One Shot Prediction")
    ax[0].set_ylabel("Humidity (percent)")
    ax[3].set_ylabel("Humidity (percent)")
    ax[6].set_ylabel("Humidity (percent)")
    ax[6].set_xlabel("Hours")
    ax[7].set_xlabel("Hours")
    ax[8].set_xlabel("Hours")
    ax[0].legend(loc=2, numpoints=1)
    plt.show()


if __name__ == '__main__':
    main()
