============
Installation
============

To install the code, we assume the user is working in a UNIX-enabled
environment. All code was tested an Ubuntu 16.04 operating system.
To download the package please take the following steps in the UNIX environment
in the order described:

1. sudo apt-get update
2. sudo apt-get install gcc
3. sudo apt-get install libcr-dev mpich mpich-doc
4. wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
5. bash Miniconda3-latest-Linux-x86_64.sh

        - Allow conda to have access to PATH (i.e., one can type conda ... and this will call the conda package manager)

6. Exit shell to allow conda to have access to PATH and log back in
7. conda install numpy
8. git clone https://github.com/zblanks/parallel_esn.git
9. cd parallel_esn
10. pip install -e .

        - This ensures that the Cython code compiles for the package

To test that the package was installed correctly, type the following command

::

        python -m parallel_esn.experiments.profile_seq --num_iter 1

In the Ubuntu 16.04 OS, you should see the following:

::

        Iteration 0
        {'k': 3, 'hidden_dim': 112, 'spectral_radius': 1.2940912584107198, 'p': 0.10919073963823023, 'alpha': 0.8153388906631103, 'beta': 141.20288318144654}
        validation loss = 0.616245337445537

The output may be slightly different depending on the OS, but the iteration
number, hyper-parameters, and validation loss should be printed.
