# === func-gen- : chroma/chromacpp/chromacpp fgp chroma/chromacpp/chromacpp.bash fgn chromacpp fgh chroma/chromacpp
chromacpp-src(){      echo chroma/chromacpp/chromacpp.bash ; }
chromacpp-source(){   echo ${BASH_SOURCE:-$(env-home)/$(chromacpp-src)} ; }
chromacpp-vi(){       vi $(chromacpp-source) ; }
chromacpp-env(){      elocal- ; }
chromacpp-usage(){ cat << EOU

CHROMACPP
==========

Motivation
-----------

Easy forking CUDA/Chroma propagation from the Geant4 process
to avoid overheads and config problems from 
multi process communication.

Alternate Approaches
--------------------

#. Work out how to embed python/pycuda into a C/C++ wrapper
#. g4daechroma.py : keep python/numpy/pycuda/GPUGeometry but drop the bells and whistles

Objective
----------

Explore how difficult to reimplement chroma propagation
using just: 
  
* C
* std C++
* CUDA (maybe thrust)
* npyreader code, 
* zmq
* chroma cuda c propagator

i.e. replacing the marshalling done by 

* python, pycuda, numpy, chroma GPUGeometry 
 

Challenges
------------

Getting Geometry/Material/Surface arrays onto GPU
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Write npy serialized arrays from python level and 
read these back using npyreader techniques

* chroma/chroma/npycacheable.py
* env/chroma/chromacpp/chromacpp.c

Looks to be tractable, but probably a least half a weeks work.

Handling BVH nodes data
~~~~~~~~~~~~~~~~~~~~~~~~~

Unknown, need to get to learn more about BVH creation, to
see if can easily persist to NPY in similar manner to geometry.

Photon Data handling
~~~~~~~~~~~~~~~~~~~~~

Idea:

#. Create NPY byte stream in the Geant4 process and send that over the wire (instead of ROOT serialized)

   * *npyreader-* does not do this
   * *cnpy-* *cnpy.h* does
    
#. Recreate arrays and cudaMemcpy them to GPU







EOU
}
chromacpp-dir(){ echo $(env-home)/chroma/chromacpp ; }
chromacpp-cd(){  cd $(chromacpp-dir); }
chromacpp-mate(){ mate $(chromacpp-dir) ; }
chromacpp-get(){
   local dir=$(dirname $(chromacpp-dir)) &&  mkdir -p $dir && cd $dir

}


chromacpp-build(){

   chromacpp-cd

   clang chromacpp.c -g -c
   clang npyreader.c -g -c

   clang *.o -o chromacpp

}

