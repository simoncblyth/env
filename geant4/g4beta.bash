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

INSTALLS 
----------

N

Geant4 has been pre-configured to look for datasets
in the directory: /data1/env/local/env/geant4/geant4.10.00.b01.local/share/Geant4-10.0.0/data
 

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

