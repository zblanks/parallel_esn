===============================
Parallel Echo State Network
===============================

.. image:: https://img.shields.io/travis/zblanks/parallel_esn.svg
        :target: https://travis-ci.org/zblanks/parallel_esn

.. image:: https://img.shields.io/pypi/v/parallel_esn.svg
        :target: https://pypi.python.org/pypi/parallel_esn


This repository contains the code to access the Python package *parallel_esn* which was developed for the Harvard CS205 final project. The code is under a three clause BSD license. To read the documentation for the package code visit: 

* Documentation: https://zblanks.github.io/parallel_esn.

Code Description
----------------
The source code used to develop the package as well as run experiments for the project are contained in the /parallel_esn folder. This directory has the following structure:

::

        parallel_esn/
        │   bo.py
        │   esn.py
        │   progress.py
        │   train_esn.pyx
        │   utils.py
        │   _version.py
        │   __init__.py
        │
        ├───data/
        ├───example/
        ├───experiments/
        │       mpi_west_coast_weather.py
        │       profile_mpi.py
        │       profile_seq.py
        │       seq_west_coast_weather.py
        │       __init__.py
        │
        ├───tests/
       
The folders and files of note are as follows:

* bo.py: Defines the class for the Bayesian optimization used to train the echo state network (ESN)
* esn.py: Defines the class used to train and validate an echo state network
* train_esn.pyx: Cython code used to optimize part of the training process
* data/: Contains an example data set used to check the correctness of the implementation
* example/: A number of small scale examples to demonstrate versatility and correctness of implementation
* tests/: Unit tests used to validate the functions defined in the package
* seq_west_coast_weather.py: Sequential implementation of the large-scale ESN training and hyper-parameter search
* mpi_west_coast_weather.py: Hybrid MPI-OpenMP implementation of the large-scale ESN training and hyper-parameter search

        - Note: MPI is explictly defined, but OpenMP is controlled via environment variables (more details to follow)

Installation
------------
To install the code, we assume the user is working in a UNIX-enabled environment. All code was tested an Ubuntu 16.04 operating system. To download the package please take the following steps in the UNIX environment in the order described:

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

The output may be slightly different depending on the OS, but the iteration number, hyper-parameters, and validation loss should be printed.

Running Experiments
-------------------
The experimental results detailed at https://rednotion.github.io/parallel_esn_web/ were obtained by working on an MPI cluster of m4.2xlarge AWS instances.

Getting Data for Experiments
============================
To get the data that was used for the large-scale sequential and parallel experiments, type the following command:

::

        wget https://raw.githubusercontent.com/rednotion/parallel_esn_web/master/west_coast_weather.csv

This is the data that is used for the sequential and parallel experiments detailed below.

Sequential Experiment
=====================
To run the sequential version of the experiment, type the following commands

::

        export OPENBLAS_NUM_THREADS=1
        mkdir figs
        python -m parallel_esn.experiments.seq_west_coast_weather
        
The number of threads is explictly set to one because NumPy, the package used to perform matrix operations in the code, will by default use as many threads as it can find for certain operations such as matrix multiplication. Therefore to ensure that you are running the sequential version this environment variable must be set. Additionally, create the directory figs/ because the code will generate plots which show the test set results from training and validating the ESN. The sequential code has the following command line arguments which can be passed to the module

* --num_iter: Number of Bayesian optimization iterations; default = 1
* --filename: Filename used to call the west coast weather data; default = west_coast_weather.csv
* --outdir: Directory to save the generated figures; default = figs
* --verbose: Controls the verbosity of the training process -- valid values are {0, 1}; default = 0

Hybrid Parallel Experiment
==========================
To run the hybrid parallel version of the experiment, type the following commands

::
        
        export OPENBLAS_NUM_THREADS=n
        mkdir figs-mpi
        mpirun -np x python -m parallel_esn.experiments.mpi_west_coast_weather --outdir figs-mpi
        
The number of threads is explictly set using the export command; howver, again NumPy will default to all available threads if no value is provided. The MPI application is controlled through standard MPI commands such as the number of tasks and, if there are multiple nodes, by providing the list of hosts. The command line arguments passed to the module are the same as the sequential version of the code.
