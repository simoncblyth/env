# setup.py:

import os

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = [Extension("test_recarray", ["test_recarray.pyx"])]

setup(
  name = 'record array test',
  cmdclass = {'build_ext':build_ext},
  include_dirs=[os.path.expandvars("$PYTHON_SITE/numpy/core/include")],
  ext_modules = ext_modules
)


