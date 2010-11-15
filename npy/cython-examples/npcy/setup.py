#!/usr/bin/env python
"""
      python setup.py build_ext --inplace
"""

import os

from distutils.core import setup 
from distutils.extension import Extension 
from Cython.Distutils import build_ext 

#import distutils.sysconfig as conf
#os.environ.update(PYTHON_SITE=conf.get_python_lib())

import numpy as np


# $PYTHON_SITE/numpy/numarray/include

ext_modules = [
     Extension("npcy", 
                ["npcy.pyx"],
                include_dirs=[np.get_include()],
             )] 

setup( 
         name = "Numpy Cython", 
     cmdclass = {'build_ext': build_ext}, 
   ext_modules = ext_modules 
) 

