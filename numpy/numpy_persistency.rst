
NumPy Persistancy
===================

.. contents:: :local:

NPY 
----

Simple uncompressed single ndarray format read/written by `np:load/np:save` 

* https://github.com/numpy/numpy/blob/master/numpy/lib/format.py
* https://github.com/numpy/numpy/blob/master/numpy/lib/npyio.py
* https://github.com/numpy/numpy/blob/master/doc/neps/npy-format.rst

CNPY : NPY from C++
~~~~~~~~~~~~~~~~~~~~~

* https://github.com/rogersce/cnpy

* cnpy lets you read and write NPY and NPZ formats in C++
* lightweight: 2 files, MIT license

npyreader : NPY from C
~~~~~~~~~~~~~~~~~~~~~~~~~~

* see *npyreader-*

* http://jcastellssala.wordpress.com/2014/02/01/npy-in-c/
* http://sourceforge.net/projects/kxtells.u/files/

np:savez
~~~~~~~~~

* http://stackoverflow.com/questions/22356612/numpy-savez-load-thousands-of-arrays-but-not-in-one-step

savez in the current numpy is just writing the arrays to temp files with
numpy.save, and then adds them to a zip file (with or without compression).

If you're not using compression, you might as well skip step 2 and just save
your arrays one by one, and keep all 4000 of them in a single folder.

NPZ is slow+inconvenient
~~~~~~~~~~~~~~~~~~~~~~~~~~~

numpy source and searches suggests npz (which packs multiple arrays into a single uncompressed zip container) 
has a very slow temporary file writing implementation.  Speed not such a problem for 
initialization of the propagator, but might as well avoid learning to navigate zip files in
C++ and instead just learn directory traversal.

JobLib : pure python, using numpy and pickle 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* http://gael-varoquaux.info/blog/?p=159
* Pickle subclass override, that adds separate NumPy handling 
* https://github.com/joblib/joblib/blob/master/joblib/numpy_pickle.py


cPickle / Pickling Tools
~~~~~~~~~~~~~~~~~~~~~~~~~~~

* http://stackoverflow.com/questions/1296162/how-can-i-read-a-python-pickle-database-file-from-c
* http://www.picklingtools.com/


HDF5
~~~~~

Too heavyweight

* http://www.h5py.org/







