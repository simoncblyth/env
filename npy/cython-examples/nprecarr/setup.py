# setup.py:

import os

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import numpy as np


ext_modules =  [ Extension("recarray", ["recarray.pyx"], include_dirs=[np.get_include()] )]

setup(
  name = 'record array test',
  cmdclass = {'build_ext':build_ext},
  ext_modules = ext_modules
)


