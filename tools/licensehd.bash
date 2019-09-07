licensehd-source(){   echo ${BASH_SOURCE} ; }
licensehd-edir(){ echo $(dirname $(licensehd-source)) ; }
licensehd-ecd(){  cd $(licensehd-edir); }
licensehd-dir(){  echo $LOCAL_BASE/env/tools/licensehd ; }
licensehd-cd(){   cd $(licensehd-dir); }
licensehd-vi(){   vi $(licensehd-source) $(licensehd-dir)/*.py ; }
licensehd-env(){  elocal- ; }
licensehd-usage(){ cat << EOU

LicenseHD
============

Started from https://github.com/johann-petrak/licenseheaders
but ended up rolling my own due to problems with stomping of 
preexisting headers and a code structure that made debugging
difficult.

Features
--------

* Foreign license headers are detected and left untouched.
* Update all Copyright lines with one command
* Can run on individual paths or sub-directories for debugging/testing  

Usage
-----

Prepend headers to all files with handled extensions::

   licensehd--

Update all headers with a new year range::

   licensehd-update



Unchanged excluding .rst .txt .in .cmake mostly have their own header or are trivials
----------------------------------------------------------------------------------------

::

    [blyth@localhost opticks]$ hg st -c  | grep -v .rst | grep -v .txt | grep -v .in | grep -v .cmake
    C .hgignore
    C LICENSE
    C Makefile
    C cfg4/C4Cerenkov1042.cc
    C cfg4/C4Cerenkov1042.hh
    C cfg4/CTestDetector.cc.old
    C cfg4/Cerenkov.hh
    C cfg4/DsG4OpBoundaryProcess.cc
    C cfg4/DsG4OpBoundaryProcess.h
    C cfg4/G4Cerenkov1042.cc
    C cfg4/G4Cerenkov1042.hh
    C cfg4/OpRayleigh.hh
    C examples/Geant4/OpNovice/GNUmakefile
    C examples/Geant4/OpNovice/History
    C examples/Geant4/OpNovice/OpNovice.cc
    C examples/Geant4/OpNovice/OpNovice.err
    C examples/Geant4/OpNovice/OpNovice.out
    C examples/Geant4/OpNovice/README
    C examples/Geant4/OpNovice/gui.mac
    C examples/Geant4/OpNovice/icons.mac
    C examples/Geant4/OpNovice/optPhoton.mac
    C examples/Geant4/OpNovice/run.png
    C examples/Geant4/OpNovice/src/OpNoviceActionInitialization.cc
    C examples/Geant4/OpNovice/src/OpNoviceDetectorConstruction.cc
    C examples/Geant4/OpNovice/src/OpNovicePhysicsList.cc
    C examples/Geant4/OpNovice/src/OpNovicePhysicsListMessenger.cc
    C examples/Geant4/OpNovice/src/OpNovicePrimaryGeneratorAction.cc
    C examples/Geant4/OpNovice/src/OpNovicePrimaryGeneratorMessenger.cc
    C examples/Geant4/OpNovice/src/OpNoviceRunAction.cc
    C examples/Geant4/OpNovice/vis.mac
    C examples/ThrustOpenGLInterop/SKIP
    C examples/UseCSGBSP/SKIP
    C examples/UseG4DAE/SKIP
    C examples/UseYoctoGLRap/SKIP
    C examples/cudarap
    C examples/thrustrap
    C extg4/OpNoviceDetectorConstruction.cc
    C extg4/OpNoviceDetectorConstruction.hh
    C notes/Makefile
    C npy/PyMCubes/LICENSE
    C oglrap/gleq.h
    C oglrap/old_gleq.h
    C oglrap/tests/gleq_check.c
    C optixrap/cu/helpers.h
    C tests/tboolean.bash.dead


EOU
}
licensehd-get(){
   local dir=$(dirname $(licensehd-dir)) &&  mkdir -p $dir && cd $dir

   [ -d licensehd ] && return 
   git clone git@github.com:simoncblyth/licensehd.git
   chmod ugo+x licensehd/licensehd.py 
}

licensehd-owner(){    echo Opticks Team ; }
licensehd-projdir(){  echo $HOME/opticks ; }
licensehd-projname(){ echo Opticks ; }
licensehd-projurl(){  echo https://bitbucket.org/simoncblyth/opticks ; }
licensehd-tmpl(){     echo under-apache-2 ; }
licensehd-years(){    echo ${LICENSEHD_YEARS:-2019} ; }

licensehd-export(){  export PATH=$(licensehd-dir):$PATH ; }

licensehd--(){
      $(licensehd-dir)/licensehd.py \
                --projdir $(licensehd-projdir) \
                --owner "$(licensehd-owner)" \
                --projname "$(licensehd-projname)" \
                --projurl "$(licensehd-projurl)" \
                --tmpl $(licensehd-tmpl) \
                --years $(licensehd-years) \
                 $*

}

licensehd-update(){ LICENSEHD_YEARS="2019-$(date +%Y)" licensehd-- --update ; }

licensehd-revert(){
   local iwd=$(pwd)
   cd $(licensehd-projdir)

   echo before
   hg st .

   hg revert --all --no-backup
   find . -name '*.tmp' -exec rm -f {} \;
   find . -name '*.orig' -exec rm -f {} \;

   echo after 
   hg st .

   cd $iwd
}

licensehd-info(){ cat << EOI

   licensehd-owner    : $(licensehd-owner)
   licensehd-projdir  : $(licensehd-projdir)
   licensehd-projname : $(licensehd-projname)
   licensehd-projurl  : $(licensehd-projurl)
   licensehd-tmpl     : $(licensehd-tmpl)
   licensehd-years    : $(licensehd-years)

EOI
}



