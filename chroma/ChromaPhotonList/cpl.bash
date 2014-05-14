# === func-gen- : chroma/cpl fgp chroma/cpl.bash fgn cpl fgh chroma
cpl-src(){      echo chroma/ChromaPhotonList/cpl.bash ; }
cpl-source(){   echo ${BASH_SOURCE:-$(env-home)/$(cpl-src)} ; }
cpl-vi(){       vi $(cpl-source) ; }
cpl-usage(){ cat << EOU

CHROMAPHOTONLIST
==================

#. could remove Geant4 dependency with very little pain by changing interface to 
   collect floats and int
#. ROOT dependency more difficult, due to need for TObject serialization


PyROOT/numpy interfacing
----------------------------------

* https://github.com/rootpy/rootpy
* http://rootpy.github.io/root_numpy/

From PyROOT
------------

::

    In [3]: ROOT.gSystem.Load("/usr/local/env/chroma/ChromaPhotonList/lib/libChromaPhotonList.dylib")
    Out[3]: 0

    In [4]: cpl = ROOT.ChromaPhotonList()

    In [5]: cpl
    Out[5]: <ROOT.ChromaPhotonList object ("ChromaPhotonList") at 0x7ffa059c0ed0>

    In [6]: cpl.x
    Out[6]: <ROOT.vector<float> object at 0x7ffa059c0ee0>

    In [7]: cpl.x.push_back(1)

    In [8]: cpl.x.size()
    Out[8]: 1L



EOU
}
cpl-dir(){  echo $(local-base)/env/chroma/ChromaPhotonList ; }
cpl-sdir(){ echo $(env-home)/chroma/ChromaPhotonList ; }
cpl-bdir(){ echo /tmp/env/chroma/ChromaPhotonList ; }

cpl-cd(){   cd $(cpl-sdir); }
cpl-icd(){  cd $(cpl-dir); }
cpl-scd(){  cd $(cpl-sdir); }
cpl-bcd(){  cd $(cpl-bdir); }

cpl-verbose(){ echo 1 ; }
cpl-prefix(){ echo $(cpl-dir) ; }

cpl-geant4-home(){ 
  case $NODE_TAG in 
    D) echo /usr/local/env/chroma_env/src/geant4.9.5.p01 ;;
  esac
}
cpl-geant4-dir(){ 
  case $NODE_TAG in 
    D) echo /usr/local/env/chroma_env/lib/Geant4-9.5.1 ;;
  esac
}
cpl-rootsys(){
  case $NODE_TAG in 
    D) echo /usr/local/env/chroma_env/src/root-v5.34.14 ;;
  esac
}

cpl-lib(){ echo $(cpl-prefix)/lib/libChromaPhotonList.dylib ; }

cpl-env(){      
   elocal- 
   export GEANT4_HOME=$(cpl-geant4-home)
   export ROOTSYS=$(cpl-rootsys)   # needed to find rootcint for dictionary creation
}

cpl-export(){
   export CHROMAPHOTONLIST_LIB=$(cpl-lib)
   export CHROMAPHOTONLIST_PREFIX=$(cpl-dir)
}

cpl-ipython(){
   CHROMAPHOTONLIST_LIB=$(cpl-lib) ipython.sh $(cpl-sdir)/cpl.py 
}


cpl-wipe(){
   local msg="=== $FUNCNAME :"
   local bdir="$(cpl-bdir)"
   echo $msg deleting bdir $bdir
   rm -rf "$bdir"
}
cpl-cmake(){
   type $FUNCNAME
   local iwd=$PWD
   mkdir -p $(cpl-bdir)   
   cpl-bcd
   cmake -DGeant4_DIR=$(cpl-geant4-dir) \
         -DCMAKE_INSTALL_PREFIX=$(cpl-prefix) \
         $(cpl-sdir) 

   cd $iwd
}
cpl-make(){
   local iwd=$PWD
   cpl-bcd
   make $* VERBOSE=$(cpl-verbose) 
   cd $iwd
}
cpl-install(){
   cpl-make install
}
cpl-build(){
   cpl-cmake
   cpl-make
   cpl-install
}
cpl-build-full(){
   cpl-wipe
   cpl-build
}
cpl-otool(){
   otool-
   otool-info $(cpl-lib)
}


cpl-nuwapkg(){    echo $DYB/NuWa-trunk/dybgaudi/Utilities/Chroma ; }  
cpl-nuwapkg-cd(){ cd $(cpl-nuwapkg) ; }
cpl-nuwapkg-cpto(){

   local pkg=$(cpl-nuwapkg)   
   local nam=$(basename $pkg)
   local inc=$pkg/$nam
   local src=$pkg/src
   local dict=$pkg/dict
  
   local iwd=$PWD 
   cpl-scd

   cp ChromaPhotonList.hh    $inc/
   cp ChromaPhotonList.cc    $src/
   cp ChromaPhotonList_LinkDef.h $dict/

   perl -pi -e 's,ChromaPhotonList.hh,Chroma/ChromaPhotonList.hh,' $src/ChromaPhotonList.cc 


   cd $iwd
}




