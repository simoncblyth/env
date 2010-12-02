#!/usr/bin/env python
"""
     Cython based extensions providing fast 
     creation of numpy record arrays from mysql queries

     Build with : 

         rm *.{c,so,pyc} ; python setup.py build_ext -i

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

import _mysql
#assert _mysql.__version__ == "1.2.3" 


def ExtArgs( **kwargs ):
    kwargs.update( extra_compile_args=mysql_config("cflags").split(), 
                      extra_link_args=mysql_config("libs").split(),)
    kwargs.setdefault('include_dirs',[]).extend( mysql_config("include")[2:] )
    return kwargs 



if _mysql.__version__ == "1.3.0":   ## unreleased
    ext_modules = [ 
        Extension( "c_api",     ["c_api.pyx"],     **ExtArgs() ),
        Extension( "dcspmthv",  ["dcspmthv.pyx"],  **ExtArgs( include_dirs=[ np.get_include(), _mysql_inc, ".", ],)),
        Extension( "npy",       ["npy.pyx"],       **ExtArgs( include_dirs=[ np.get_include(), _mysql_inc, ".", ],)),
  ]
else:
    ext_modules = [ 
        Extension( "c_api",     ["c_api.pyx"],     **ExtArgs() ),
        Extension( "dcspmthv",  ["dcspmthv.pyx"],  **ExtArgs( include_dirs=[ np.get_include(), _mysql_inc, ".", ],)),
        Extension( "npy",       ["npy.pyx"],       **ExtArgs( include_dirs=[ np.get_include(), "." ],)),
  ]



setup(
  name = 'npmy',
  #packages = [ 'npmy', 'mysql', 'mysql.api', 'mysql.npy', ], 
  cmdclass = {'build_ext':build_ext},
  ext_modules = ext_modules ,
)


