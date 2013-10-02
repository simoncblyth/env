# === func-gen- : geant4/g4beta fgp geant4/g4beta.bash fgn g4beta fgh geant4
g4beta-src(){      echo geant4/g4beta.bash ; }
g4beta-source(){   echo ${BASH_SOURCE:-$(env-home)/$(g4beta-src)} ; }
g4beta-vi(){       vi $(g4beta-source) ; }
g4beta-env(){      elocal- ; }
g4beta-usage(){ cat << EOU

Geant4 Beta 
===========

* http://geant4.kek.jp/geant4/support/Beta4.10.0-1.txt
* http://geant4.web.cern.ch/geant4/UserDocumentation/UsersGuides/InstallationGuide/BackupVersions/V9.4/html/ch04s02.html

cmake G4 build is a big improvement, a do nothing rebuild now takes seconds.


INSTALLS 
----------

N

Geant4 has been pre-configured to look for datasets
in the directory: /data1/env/local/env/geant4/geant4.10.00.b01.local/share/Geant4-10.0.0/data
 
C

No cmake in the distro repo



GDML
~~~~~~

Needs xerces-c::

    sudo yum --enablerepo=epel install xerces-c
    sudo yum --enablerepo=epel install xerces-c-devel





EOU
}
g4beta-dir(){ echo $(local-base)/env/geant4/$(g4beta-name) ; }
g4beta-bdir(){ echo $(g4beta-dir).build ; }
g4beta-idir(){ echo $(g4beta-dir).local ; }
g4beta-cd(){  cd $(g4beta-dir); }
g4beta-bcd(){  cd $(g4beta-bdir); }
g4beta-name(){ echo geant4.10.00.b01 ; }
g4beta-url(){ echo http://geant4.cern.ch/support/source/$(g4beta-name).tar.gz ; }
g4beta-get(){
   local dir=$(dirname $(g4beta-dir)) &&  mkdir -p $dir && cd $dir
   local url=$(g4beta-url)
   local tgz=$(basename $url)
   local nam=${tgz/.tar.gz}
   [ ! -f "$tgz" ] && curl -L -O $url
   [ ! -d "$nam" ] && tar zxvf $tgz

   mkdir -p $(g4beta-bdir)
}

g4beta-cmake(){
   cd $(g4beta-bdir)
   cmake -DCMAKE_INSTALL_PREFIX=$(g4beta-idir) -DGEANT4_INSTALL_DATA=ON -DGEANT4_USE_GDML=ON ../$(g4beta-name) 
}

g4beta-make(){
   cd $(g4beta-bdir)
   make
}


g4beta-plat(){ echo i686-slc5-gcc41-dbg ; }
g4beta-boost-idir(){ echo $DYB/external/Boost/1.38.0_python2.7/$(g4beta-plat) ; } 
g4beta-xercesc-idir(){ echo $DYB/external/XercesC/2.8.0/$(g4beta-plat) ; }
g4beta-python-idir(){ echo $DYB/external/Python/2.7/$(g4beta-plat) ; }

g4beta-pyconfigure(){
   cd $(g4beta-dir)
   ./configure  linux \
                  --with-g4install-dir=$(g4beta-idir)
                --with-boost-incdir=$(g4beta-boost-idir)/include/boost-1_38 \
                --with-boost-libdir=$(g4beta-boost-idir)/lib \
                --with-xercesc-incdir=$(g4beta-xercesc-idir)/include \
                --with-xercesc-libdir=$(g4beta-xercesc-idir)/lib \
                --with-python-incdir=$(g4beta-python-idir)/include/python2.7 \
                --with-python-libdir=$(g4beta-python-idir)/lib \
                --with-boost-python-lib=boost_python-gcc41-mt 

}


g4beta-diff(){
   nuwa-

   local def="source/visualization/management/include/G4VSceneHandler.hh"
   local path=${1:-$def}
   local cmd="diff $(nuwa-g4-bdir)/$path $(g4beta-dir)/$path"
   echo $cmd
   eval $cmd
}


