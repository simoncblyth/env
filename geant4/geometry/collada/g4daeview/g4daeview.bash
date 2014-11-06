# === func-gen- : geant4/geometry/collada/g4daeview/g4daeview fgp geant4/geometry/collada/g4daeview/g4daeview.bash fgn g4daeview fgh geant4/geometry/collada/g4daeview
g4daeview-src(){      echo geant4/geometry/collada/g4daeview/g4daeview.bash ; }
g4daeview-source(){   echo ${BASH_SOURCE:-$(env-home)/$(g4daeview-src)} ; }
g4daeview-vi(){       vi $(g4daeview-source) ; }
g4daeview-env(){      
    elocal- 
    chroma-
}
g4daeview-usage(){ cat << EOU

DAEVIEW FUNCTIONS
==================

*g4daeview*
         launch app

*g4daeview-ctl*
         send UDP message to app


DEPENDENCIES
-------------

Top level packages required:

* numpy
* pyopengl
* glumpy
* pycollada
* env.geant4.geometry.collada.daenode

Optional packages:

* chroma  (requires pycuda, CUDA and suitable NVIDIA GPU compute capability 3+ Kepler series or later)

  * env.cuda.cuda_launch  

COLLADA geometry files, typically 3 files (for DayaBay, Lingao and Far) each being of 6-8 MB::
 
    env-
    export-
    export-get-all

Define envvar pointing at chosen geometry file::

   export DAE_NAME=/usr/local/env/geant4/geometry/export/DayaBay_VGDX_20140414-1300/g4_00.dae




Delta : OSX 10.9.2 chroma virtualenv python, based off macports py27
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#. development machine

G4PB : OSX 10.5.8 macports py26 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#. fallback from OrderedDict to dict, means help and config listings will be in scrambled order 
#. had to kludge glumpy glut main loop, see *glumpy-vi* 
#. keyboard/trackpad binding issue


EOU
}
g4daeview-dir(){ echo $(env-home)/geant4/geometry/collada/g4daeview; }
g4daeview-cd(){  cd $(g4daeview-dir); }
g4daeview-mate(){ mate $(g4daeview-dir) ; }




g4daeview(){
    g4daeview.sh $*  
}
g4daeview-ctl(){
    udp.py "$*"
}


g4daeview-transmogrify(){
   echo -n 
   local src=$ENV_HOME/geant4/geometry/collada/g4daeview
   local tgt=/tmp/g4daeview

   mkdir -p $tgt $tgt/doc $tgt/g4daeview $tgt/bin 

   local path
   local tpath
   local cmd

   ls -1 $src/*.rst | while read path ; do
      tpath=$tgt/doc/$(basename $path) 
      cmd="cp $path $tpath"  
   done

   ls -1 $src/*.py | while read path ; do
      tpath=$tgt/g4daeview/$(basename $path) 
      cmd="cp $path $tpath"  
      [ $path -nt $tpath ] && echo $cmd && eval $cmd
   done

}

g4daeview-transmogrify-notes(){ cat << EON

Making g4daeview.py more portable
==================================

Pulling g4daeview sources from env to transmogrify 
into a typical python package layout  
in a separate repo called g4daeview

::

  g4daeview/
     g4daeview/..     repeat repo name for top python module 
     src/             non-python sources 
     doc/             Sphinx rst sources
     cuda/            follow chroma conventions
     bin/
     setup.py
     README.rst       for bitbucket intro

Issues: 

* many python imports will need to be changed 
* CUDA split off ? 
* quite a few dependencies 
* live config for ZMQ connection
* want to auto-optionalize dependencies

Required Dependencies
---------------------- 

* numpy, glumpy, pyopengl, pycollada

Strategy
---------

#. rejig in place to compartmentalize dependencies
   before effecting partition into a new repo ?


Optional Dependency Silos
--------------------------

standalone functionality degrade
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* PIL for screen grabs

photon transport/serialization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Functionality could in principle be used without CUDA/chroma, 
for example to allow propagation presentation of a pre-propagated VBO 
on a non-CUDA node/device. 

* env.chroma.ChromaPhotonList.cpl 

  * ZMQ
  * ROOT : only used for TObject (de)serialization,
    limits portability substantially

TODO:Find ROOT Alternative
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Need focussed AND easily portable (in this context means C/C++) 
compression/decompression project like capnproto
      
* http://kentonv.github.io/capnproto/

CUDA/chroma et al
~~~~~~~~~~~~~~~~~~~

* Inevitable portability breaker
* need to compartmentalise to ensure all the functionality that 
  can work without it can be portable.

* "hats" include daechromacontext.py  

* pycuda.gl.autoinit  
* env.cuda.cuda_launch  
* chroma.gpu.tools
* chroma.gpu.geometry



EON
}

