from os import path
from setuptools import setup, find_packages, Extension
import sys
import versioneer
import numpy as np


# NOTE: This file must remain Python 2 compatible for the foreseeable future,
# to ensure that we error out properly for people with outdated setuptools
# and/or pip.
min_version = (3, 6)
if sys.version_info < min_version:
    error = """
parallel_esn does not support Python {0}.{1}.
Python {2}.{3} and above is required. Check your Python version like so:

python3 --version

This may be due to an out-of-date pip. Make sure you have pip >= 9.0.1.
Upgrade pip like so:

pip install --upgrade pip
""".format(*(sys.version_info[:2] + min_version))
    sys.exit(error)

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.rst'), encoding='utf-8') as readme_file:
    readme = readme_file.read()

with open(path.join(here, 'requirements.txt')) as requirements_file:
    # Parse requirements.txt, ignoring any commented-out lines.
    requirements = [line for line in requirements_file.read().splitlines()
                    if not line.startswith('#')]


setup(
    name='parallel_esn',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="Parallel Echo State Networks for Harvard CS205 final project",
    long_description=readme,
    author="Zachary Blanks",
    author_email='zachblanks17@gmail.com',
    url='https://github.com/zblanks/parallel_esn',
    python_requires='>={}'.format('.'.join(str(n) for n in min_version)),
    packages=find_packages(exclude=['docs', 'tests']),
    ext_modules=[Extension('parallel_esn.train_esn',
                           ['parallel_esn/train_esn.pyx'],
                           include_dirs=[np.get_include()])],
    setup_requires=['setuptools>=18.0', 'cython>=0.27'],
    entry_points={
        'console_scripts': [
            # 'some.module:some_function',
            ],
        },
    include_package_data=True,
    package_data={
        'parallel_esn': [
            # When adding files here, remember to update MANIFEST.in as well,
            # or else they will not be included in the distribution on PyPI!
            'data/*.csv',
            ]
        },
    install_requires=requirements,
    license="BSD (3-clause)",
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
    ],
)
