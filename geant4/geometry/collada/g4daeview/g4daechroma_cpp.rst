G4DAEChroma.cpp : ZMQResponder, CUDA kernel call  
====================================================


Planned C++ implementation of `g4daechroma.py` with few dependencies 
that just collects photons bytes streams, 
deserializes into arrays, copies to GPU, runs kernel, 
copy back, serialize and reply.


Progress
---------

* `chromacpp-` prelim look at directory traversal and reading npy files
* `--geocache` is now operational, so have the cache 



Motivation
-----------

* installation simplicity, just a few Chroma/CUDA kernel calls 
  and copying vectors, 

* convenience of forking from Geant4 process to run the 
  GPU propagation on same machine at the generation

* speed

Dependencies
-------------

* CUDA (+thrust?)
* ZMQ
* (de)serialization

Initialization
----------------

#. load geometry/materials/surface data from some persisted/cached format 
#. copy geometry/materials/surface data to GPU building GPU structs
   holding the GPU pointers from the copies 

Propagation
--------------

#. ZMQ poll for bytes
#. deserialize bytes into photon data objects of some type 

   * container object with std::vector<float> members 
   * container with numpy array members
   * single numpy array, using union int/float trickery for simple decoding from C++

#. copy arrays to GPU 
#. kernel call
#. get arrays back to CPU
#. recompose transport object with propagated photons
#. serialize
#. ZMQ reply  

#. avoid intermediate `ChromaPhotonList` and instead load 
   photons into the transport class


NPY Geometry Cache
--------------------

* see :doc:`/numpy/numpy_persistency`

Do not want to recreate pycollada parsing again in C++, 
so need so do that in python and persist the gleaned data in numpy 
NPY serialization format, which can be read from C/C++.

Intermediate geometry cache format that is written by 
the graphical `g4daeview.py` and can 
be used as an initialization speedup for the viewer. 

Performance not an major issue for initialization, 
so just use NPY as convenient to be written from python/numpy 
and read in C++ with CNPY.

Use directory structure with single npy files::

    materials/000/absorption_length.npy
                  reemission_probability.npy
                  ...
              001/absorption_length.npy
                  ...

    surfaces/001/...

    geometry/000/vertices.npy 
                 triangles.npy 



Potential Dependencies
------------------------

CuPP 
^^^^^

#. http://www.plm.eecs.uni-kassel.de/CuPP/ 

   * facilitating CUDA from C++, like pycuda does from python ?
   * probably unnecessary 

boost::program_options
^^^^^^^^^^^^^^^^^^^^^^^^^

* http://stackoverflow.com/questions/7399688/cuda-with-boost
* maybe unnecessary, handle config via bash script, envvars  


`boost::numeric::ublas::vector<double>`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* http://stackoverflow.com/questions/9059774/boostnumericublasvector-internal-data-storage-pointer

::

    boost::numeric::ublas::vector<double> vector;
    double* ptr = &vector[0];


    vector<double, unbounded_array<double,n_elements>> vector;
    cudaMemcpy(device_dest, 
               vector.data().begin(), 
               vector.data().size(), 
               cudaMemcpyHostToDevice);


cuda::thrust
^^^^^^^^^^^^^^

* comes with CUDA, so not really an extra dependency 
* http://docs.nvidia.com/cuda/thrust/

  * allows to hide the memcpy

::

    // Copy host_vector H to device_vector D 
    thrust::device_vector<int> D = H; 


CUDA Unified Memory 
^^^^^^^^^^^^^^^^^^^^

* needs CUDA 6, compute capability 3, MBP should be capable of this
* http://devblogs.nvidia.com/parallelforall/unified-memory-in-cuda-6/


Other Libs
------------

#. pyublas : integrate Boost.UBlas and Boost.Python

   * http://documen.tician.de/pyublas/
   * allows to fill arrays in C++ that can be viewed 
     as numpy arrays at python level off the same data, **NO COPYING**

   * what about serialization ?


#. Boost-python

   * https://github.com/abingham/boost_python_tutorial
   * http://www.quora.com/How-do-I-convert-C++-vector-to-NumPy-array-using-Boost-Python



