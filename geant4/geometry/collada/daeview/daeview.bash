# === func-gen- : geant4/geometry/collada/daeview/daeview fgp geant4/geometry/collada/daeview/daeview.bash fgn daeview fgh geant4/geometry/collada/daeview
daeview-src(){      echo geant4/geometry/collada/daeview/daeview.bash ; }
daeview-source(){   echo ${BASH_SOURCE:-$(env-home)/$(daeview-src)} ; }
daeview-vi(){       vi $(daeview-source) ; }
daeview-env(){      
    elocal- 
    chroma-
}
daeview-usage(){ cat << EOU

DAEVIEW FUNCTIONS
==================

*daeview*
         launch app

*daeview-ctl*
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
daeview-dir(){ echo $(env-home)/geant4/geometry/collada/daeview; }
daeview-cd(){  cd $(daeview-dir); }
daeview-mate(){ mate $(daeview-dir) ; }

daeview(){
    daeviewgl.py $*   
}
daeview-ctl(){
    udp.py "$*"
}


