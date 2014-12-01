# === func-gen- : chroma/chromacpp/chromacpp fgp chroma/chromacpp/chromacpp.bash fgn chromacpp fgh chroma/chromacpp
chromacpp-src(){      echo chroma/chromacpp/chromacpp.bash ; }
chromacpp-source(){   echo ${BASH_SOURCE:-$(env-home)/$(chromacpp-src)} ; }
chromacpp-vi(){       vi $(chromacpp-source) ; }
chromacpp-usage(){ cat << EOU

CHROMACPP
==========

Motivation
-----------

Easy forking CUDA/Chroma propagation from the Geant4 process
to avoid overheads and config problems from 
multi process communication.

Next
------

primitives
~~~~~~~~~~~~

Mostly not used but channel_id_to_channel_index definitely is::

    delta:chroma_geometry blyth$ find . -name __primitives.dict -exec bash -c "cat {} && echo" \;
    {'channel_id_to_channel_index': {16844801: 844, 16844802: 849, 16842755: 977, 16842756: 983, 16844805: 864, 16844806: 869, 16842753: 965, 16844808: 879, 16844809: 884, 16844810: 889, 16844811: 894, 16844812: 899, 16909314: 1365, 16844814: 909, 16844815: 914, 16844816: 919, 16844817: 924, 16844818: 929, 16844819: 934, 16844820: 939, 16844821: 944, 16844822: 949, 16844823: 954, 16844824: 959, 16844804: 859, 17109763: 2348, 16909317: 1380, 16908289: 1961, 16909318: 1385, 16909319: 1390, 16909320: 1395, 17111308: 3895, 16843785: 404, 17171457: 6224, 17172230: 6649, 16909322: 1405, 16842754: 971, 16843787: 414, 16909313: 1360, 17175558: 4694, 16909324: 1415, 16909586: 1565, 16909325: 1420, 17

    {'none_name': '__none.txt', 'primitives_name': '__primitives.dict'}
    {'none_name': '__none.txt', 'primitives_name': '__primitives.dict', 'composition': {}, 'name': '__dd__Materials__LiquidScintillator0xc2308d0', 'density': 0.0}
    {'none_name': '__none.txt', 'primitives_name': '__primitives.dict', 'composition': {}, 'name': '__dd__Materials__Air0xc032550', 'density': 0.0}
    {'none_name': '__none.txt', 'primitives_name': '__primitives.dict', 'composition': {}, 'name': '__dd__Materials__Aluminium0xc542070', 'density': 0.0}
    {'none_name': '__none.txt', 'primitives_name': '__primitives.dict', 'composition': {}, 'name': '__dd__Materials__GdDopedLS0xc2a8ed0', 'density': 0.0}


Either parse the __primitives.dict or change storage to simple ndarray layout, could represent 
the mapping in (2,nchan) shaped arrays of ints.  Remember that non-simple ndarray dtype 
would need work on the C level ndarray reader.


translate the pycuda geometry struct creation into C
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Surface, Materials, Geometry structs look tractable just 
a translation of the pycuda: 

* `chroma/chroma/gpu/detector.py`
* `chroma/chroma/gpu/geometry.py` 


BVH nodes
~~~~~~~~~~~

* BVH nodes are more of a wildcard, currently they are skipped as are 
  cPickle cached separately::

    delta:chromacpp blyth$ xxd ~/.chroma/bvh/22f6c0f473127c816368da5dcc6d1458/default
    0000000: 8002 6363 6872 6f6d 612e 6276 682e 6276  ..cchroma.bvh.bv
    0000010: 680a 4256 480a 7101 2981 7102 7d71 0328  h.BVH.q.).q.}q.(
    0000020: 550c 6c61 7965 725f 626f 756e 6473 7104  U.layer_boundsq.
    0000030: 5d71 0528 636e 756d 7079 2e63 6f72 652e  ]q.(cnumpy.core.
    0000040: 6d75 6c74 6961 7272 6179 0a73 6361 6c61  multiarray.scala
    0000050: 720a 7106 636e 756d 7079 0a64 7479 7065  r.q.cnumpy.dtype
    0000060: 0a71 0755 0269 384b 004b 0187 5271 0828  .q.U.i8K.K..Rq.(
    0000070: 4b03 5501 3c4e 4e4e 4aff ffff ff4a ffff  K.U.<NNNJ....J..
    0000080: ffff 4b00 7462 5508 0000 0000 0000 0000  ..K.tbU.........
    0000090: 8652 7109 6806 6808 5508 0100 0000 0000  .Rq.h.h.U.......
    00000a0: 0000 8652 710a 6806 6808 5508 0300 0000  ...Rq.h.h.U.....

* hmm will need to modify storage to NPY



cuda without pycuda
~~~~~~~~~~~~~~~~~~~~

#. learning needed




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

Looks to be tractable, but probably a least a weeks work.

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

ZMQ responder
~~~~~~~~~~~~~~

Listen for bytes, copy em to GPU, kernel call, copy back, reply.







EOU
}
chromacpp-dir(){ echo $(env-home)/chroma/chromacpp ; }
chromacpp-cd(){  cd $(chromacpp-dir); }
chromacpp-mate(){ mate $(chromacpp-dir) ; }
chromacpp-get(){
   local dir=$(dirname $(chromacpp-dir)) &&  mkdir -p $dir && cd $dir

}

chromacpp-env(){      elocal- ; 
   export-
   export-export
}



chromacpp-build(){
   local iwd=$PWD
   chromacpp-cd

   clang chromacpp.c -g -c
   clang npyreader.c -g -c

   clang *.o -o $LOCAL_BASE/env/bin/chromacpp

   rm *.o
   cd $iwd
}

chromacpp-cachedir(){ echo $DAE_NAME_DYB_CHROMACACHE ; }
chromacpp-ccd(){ cd $(chromacpp-cachedir) ; }

chromacpp--(){
  chromacpp-build 
  chromacpp $(chromacpp-cachedir)
}


chromacpp-find(){
   local iwd=$PWD
   cd $(chromacpp-cachedir)
   find . -name '*.npy' 
   cd $iwd
}

