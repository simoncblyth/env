from distutils.core import setup, Extension
import numpy.distutils.misc_util

# https://docs.python.org/2/extending/building.html
c_ext = Extension("_npar", 
                  sources=["_npar.c", "querydata.c"],
                  libraries=["sqlite3"],
                  include_dirs=["/opt/local/include"],   # non-portable 
                 )

setup(
    ext_modules=[c_ext],
    include_dirs=numpy.distutils.misc_util.get_numpy_include_dirs(),
)
