# setup.py:

import os

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import numpy as np

mysql_config = lambda _:os.popen("mysql_config --%s" % _ ).read().strip()

mysql_inc = mysql_config("include")[2:]
_mysql_inc = os.path.expandvars("$LOCAL_BASE/env/mysql/MySQLdb-2.0/src")  # mysql-python-;mysql-python-get;...

ext = Extension(
          "npmy", 
          ["npmy.pyx"], 
          include_dirs=[ np.get_include(), _mysql_inc, mysql_inc, ],
          extra_compile_args=mysql_config("cflags").split(),
          extra_link_args=mysql_config("libs").split(),
                   )

setup(
  name = 'npmy',
  cmdclass = {'build_ext':build_ext},
  ext_modules = [ ext ],
)


