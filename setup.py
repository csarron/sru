import os
from setuptools import setup, find_packages
import build

this_file = os.path.dirname(__file__)


def readme():
    """ Return the README text.
    """
    with open('README.md') as fh:
        return fh.read()


def get_version():
    """ Gets the current version of the package.
    """
    version_py = os.path.join(os.path.dirname(__file__), 'sru/version.py')
    with open(version_py) as fh:
        for line in fh:
            if line.startswith('__version__'):
                return line.split('=')[-1].strip() \
                    .replace('"', '').replace("'", '')
    raise ValueError('Failed to parse version from: {}'.format(version_py))


setup(
    name="sru",
    version=get_version(),
    description='Training RNNs as Fast as CNNs',
    long_description=readme(),
    keywords='deep learning rnn lstm cudnn sru fast',
    url='https://github.com/taolei87/sru',
    author='Tao Lei, Yu Zhang',
    author_email='tao@asapp.com',
    license='MIT',
    # Require cffi.
    install_requires=["cffi>=1.0.0"],
    setup_requires=["cffi>=1.0.0"],
    # Exclude the build files.
    packages=find_packages(exclude=["build"]),
    # Package where to put the extensions. Has to be a prefix of build.py.
    ext_package="",
    # Extensions to compile.
    cffi_modules=[
        os.path.join(this_file, "build.py:ffi")
    ],
)
