

Building is failing ... with missing numpy types

     cimport numpy as np  
        * import the numpy.pxd   (cython header equivalent :  Cython/Includes/numpy.pxd ) 
           * cdef extern from "numpy/arrayobject.h": 



   * numpy was installed as dependency of matplotlib by pip ...





[blyth@cms01 matplotlib]$ find
/data/env/system/python/Python-2.5.1/lib/python2.5/site-packages/numpy/ -name
'*.h'
/data/env/system/python/Python-2.5.1/lib/python2.5/site-packages/numpy/distutils/tests/swig_ext/src/zoo.h
/data/env/system/python/Python-2.5.1/lib/python2.5/site-packages/numpy/random/randomkit.h
/data/env/system/python/Python-2.5.1/lib/python2.5/site-packages/numpy/f2py/src/fortranobject.h
/data/env/system/python/Python-2.5.1/lib/python2.5/site-packages/numpy/core/include/numpy/old_defines.h
/data/env/system/python/Python-2.5.1/lib/python2.5/site-packages/numpy/core/include/numpy/__multiarray_api.h
/data/env/system/python/Python-2.5.1/lib/python2.5/site-packages/numpy/core/include/numpy/npy_common.h
/data/env/system/python/Python-2.5.1/lib/python2.5/site-packages/numpy/core/include/numpy/npy_interrupt.h
/data/env/system/python/Python-2.5.1/lib/python2.5/site-packages/numpy/core/include/numpy/noprefix.h
/data/env/system/python/Python-2.5.1/lib/python2.5/site-packages/numpy/core/include/numpy/npy_3kcompat.h
/data/env/system/python/Python-2.5.1/lib/python2.5/site-packages/numpy/core/include/numpy/oldnumeric.h
/data/env/system/python/Python-2.5.1/lib/python2.5/site-packages/numpy/core/include/numpy/ufuncobject.h
/data/env/system/python/Python-2.5.1/lib/python2.5/site-packages/numpy/core/include/numpy/npy_math.h
/data/env/system/python/Python-2.5.1/lib/python2.5/site-packages/numpy/core/include/numpy/numpyconfig.h
/data/env/system/python/Python-2.5.1/lib/python2.5/site-packages/numpy/core/include/numpy/npy_os.h
/data/env/system/python/Python-2.5.1/lib/python2.5/site-packages/numpy/core/include/numpy/arrayscalars.h
/data/env/system/python/Python-2.5.1/lib/python2.5/site-packages/numpy/core/include/numpy/utils.h
/data/env/system/python/Python-2.5.1/lib/python2.5/site-packages/numpy/core/include/numpy/ndarrayobject.h
/data/env/system/python/Python-2.5.1/lib/python2.5/site-packages/numpy/core/include/numpy/ndarraytypes.h
/data/env/system/python/Python-2.5.1/lib/python2.5/site-packages/numpy/core/include/numpy/__ufunc_api.h
/data/env/system/python/Python-2.5.1/lib/python2.5/site-packages/numpy/core/include/numpy/npy_endian.h
/data/env/system/python/Python-2.5.1/lib/python2.5/site-packages/numpy/core/include/numpy/npy_cpu.h
/data/env/system/python/Python-2.5.1/lib/python2.5/site-packages/numpy/core/include/numpy/_numpyconfig.h
/data/env/system/python/Python-2.5.1/lib/python2.5/site-packages/numpy/core/include/numpy/_neighborhood_iterator_imp.h
/data/env/system/python/Python-2.5.1/lib/python2.5/site-packages/numpy/core/include/numpy/arrayobject.h
/data/env/system/python/Python-2.5.1/lib/python2.5/site-packages/numpy/numarray/include/numpy/nummacro.h
/data/env/system/python/Python-2.5.1/lib/python2.5/site-packages/numpy/numarray/include/numpy/ieeespecial.h
/data/env/system/python/Python-2.5.1/lib/python2.5/site-packages/numpy/numarray/include/numpy/libnumarray.h
/data/env/system/python/Python-2.5.1/lib/python2.5/site-packages/numpy/numarray/include/numpy/cfunc.h
/data/env/system/python/Python-2.5.1/lib/python2.5/site-packages/numpy/numarray/include/numpy/numcomplex.h
/data/env/system/python/Python-2.5.1/lib/python2.5/site-packages/numpy/numarray/include/numpy/arraybase.h
[blyth@cms01 matplotlib]$ 
[blyth@cms01 matplotlib]$ 



   * after setup numpy include location ...


[blyth@cms01 numpy-cython]$ python setup.py build_ext --inplace
running build_ext
cythoning numpycython.pyx to numpycython.c
building 'numpycython' extension
creating build
creating build/temp.linux-i686-2.5
gcc -pthread -fno-strict-aliasing -DNDEBUG -g -O3 -Wall -Wstrict-prototypes
-fPIC
-I/data/env/system/python/Python-2.5.1/lib/python2.5/site-packages/numpy/core/include
-I/data/env/system/python/Python-2.5.1/include/python2.5 -c numpycython.c -o
build/temp.linux-i686-2.5/numpycython.o
/data/env/system/python/Python-2.5.1/lib/python2.5/site-packages/numpy/core/include/numpy/__multiarray_api.h:1188:
warning: '_import_array' defined but not used
/data/env/system/python/Python-2.5.1/lib/python2.5/site-packages/numpy/core/include/numpy/__ufunc_api.h:197:
warning: '_import_umath' defined but not used
gcc -pthread -shared build/temp.linux-i686-2.5/numpycython.o
-L/data/env/system/python/Python-2.5.1/lib -lpython2.5 -o numpycython.so
[blyth@cms01 numpy-cython]$ 





