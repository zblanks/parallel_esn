from pkg_resources import resource_filename
import numpy as np
import argparse
from ..esn import ESN
from ..utils import chunk_data
from ..bo import BO


def prep_data(windowsize):
    fname = resource_filename('parallel_esn', 'data/PJM_Load_hourly.csv')
    data = np.loadtxt(fname, delimiter=',', skiprows=1, usecols=[1])

    total_len = data.shape[0]
    val_len = total_len // 10
    train_len = total_len - val_len

    data = data - np.average(data)
    data /= np.std(data)

    train_data = data[:train_len]
    val_data = data[train_len:]

    trainU, trainY = chunk_data(train_data, windowsize, 20)
    valU, valY = chunk_data(val_data, windowsize, 20)
    return trainU, trainY, valU, valY


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--windowsize', type=int, nargs='?', default=160)
    parser.add_argument('--num_iter', type=int, nargs='?', default=5)
    args = parser.parse_args()

    trainU, trainY, valU, valY = prep_data(args.windowsize)

    bo = BO(k=(2, 15), hidden_dim=(10, 100), random_state=17)

    for _ in range(args.num_iter):
        H_space = bo.build_options()
        h_star = bo.find_best_choice(H_space)
        print(h_star)

        esn = ESN(input_dim=1, hidden_dim=h_star['hidden_dim'], output_dim=1,
                  k=h_star['k'], spectral_radius=h_star['spectral_radius'],
                  p=h_star['p'], alpha=h_star['alpha'], beta=h_star['beta'])

        val_loss = esn.train_validate(trainU, trainY, valU, valY, verbose=0)
        print("validation loss = {}".format(val_loss))

        bo.update_gpr(X=[h_star[val] for val in h_star.keys()], y=val_loss)


if __name__ == '__main__':
    main()
