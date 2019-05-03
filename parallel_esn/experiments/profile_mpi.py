from pkg_resources import resource_filename
import numpy as np
from mpi4py import MPI
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

    bo = BO(k=(2, 15), hidden_dim=(100, 200), random_state=17)

    comm = MPI.COMM_WORLD

    params = None
    res_info = None

    if comm.rank == 0:
        # Generate comm.size starting points because we do not place a prior
        # on the hyper-parameter space
        init_params = bo.find_best_choices(num_choices=comm.size-1)
        for i in range(1, comm.size):
            vals = [init_params[key][i] for key in init_params.keys()]
            params = dict(zip(init_params.keys(), vals))
            comm.send(params, dest=i)

        for i in range(args.num_iter):
            # We assume res_info is a dictionary {'params': {'': ..., }, '
            # 'error': ..., 'source': ...} where source is the worker that
            # sent the message
            res_info = comm.recv(source=MPI.ANY_SOURCE)
            val_error = res_info['error']
            params = res_info['params']
            bo.update_gpr(X=[params[val] for val in params.keys()], y=val_error)

            if i < args.num_iter - 1:
                params = bo.find_best_choices()
                comm.send(params, dest=res_info['source'])

        # The search process has completed and we will terminate MPI process
        for i in range(1, comm.size):
            comm.send(-1, dest=i)
    else:
        while True:
            params = comm.recv(source=0)
            if params == -1:
                break

            esn = ESN(input_dim=1, hidden_dim=params['hidden_dim'], output_dim=1,
                      k=params['k'], spectral_radius=params['spectral_radius'],
                      p=params['p'], alpha=params['alpha'], beta=params['beta'])

            val_loss = esn.train_validate(trainU, trainY, valU, valY,
                                          verbose=0)

            res_info = {'params': params, 'error': val_loss,
                        'source': comm.rank}
            comm.send(res_info, dest=0)


if __name__ == '__main__':
    main()
