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


INSTALLS
---------

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
    g4daeview.sh --with-chroma $*   
}
g4daeview-ctl(){
    udp.py "$*"
}


