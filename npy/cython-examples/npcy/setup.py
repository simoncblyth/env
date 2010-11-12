#!/usr/bin/env python
"""
      python setup.py build_ext --inplace
"""

import os

from distutils.core import setup 
from distutils.extension import Extension 
from Cython.Distutils import build_ext 


# $PYTHON_SITE/numpy/numarray/include

ext_modules = [
     Extension("npcy", 
                ["npcy.pyx"],
                include_dirs=[os.path.expandvars("$PYTHON_SITE/numpy/core/include")],
             )] 

setup( 
         name = "Numpy Cython", 
     cmdclass = {'build_ext': build_ext}, 
   ext_modules = ext_modules 
) 

