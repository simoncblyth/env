"""
     Cython based extensions providing fast 
     creation of numpy record arrays from mysql queries

     Build with : 

         rm *.{c,so} ; python setup.py build_ext -i

     TODO :
          add stringemnt mysql-python version check ... as the extension 
          hijacks the  _mysql.result struct 

     http://wiki.cython.org/PackageHierarchy
 
"""

import os
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import numpy as np

mysql_config = lambda _:os.popen("mysql_config --%s" % _ ).read().strip()
mysql_inc = mysql_config("include")[2:]
_mysql_inc = os.path.expandvars("$LOCAL_BASE/env/mysql/MySQLdb-2.0/src")  # mysql-python-;mysql-python-get;...


def ExtArgs( **kwargs ):
    kwargs.update( extra_compile_args=mysql_config("cflags").split(), 
                      extra_link_args=mysql_config("libs").split(),)
    kwargs.setdefault('include_dirs',[]).extend( mysql_config("include")[2:] )
    return kwargs 
    
ext_modules = [
     Extension( "api",       ["api.pyx"],  **ExtArgs() ),
     Extension( "npy",       ["npy.pyx"],  **ExtArgs( include_dirs=[ np.get_include(), _mysql_inc, ".", ],)),
  ]


setup(
  name = 'npmy',
  #packages = [ 'npmy', 'mysql', 'mysql.api', 'mysql.npy', ], 
  cmdclass = {'build_ext':build_ext},
  ext_modules = ext_modules ,
)


