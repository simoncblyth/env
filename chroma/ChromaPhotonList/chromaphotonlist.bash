# === func-gen- : chroma/chromaphotonlist fgp chroma/chromaphotonlist.bash fgn chromaphotonlist fgh chroma
chromaphotonlist-src(){      echo chroma/ChromaPhotonList/chromaphotonlist.bash ; }
chromaphotonlist-source(){   echo ${BASH_SOURCE:-$(env-home)/$(chromaphotonlist-src)} ; }
chromaphotonlist-vi(){       vi $(chromaphotonlist-source) ; }
chromaphotonlist-usage(){ cat << EOU

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
chromaphotonlist-dir(){  echo $(local-base)/env/chroma/ChromaPhotonList ; }
chromaphotonlist-sdir(){ echo $(env-home)/chroma/ChromaPhotonList ; }
chromaphotonlist-bdir(){ echo /tmp/env/chroma/ChromaPhotonList ; }

chromaphotonlist-cd(){   cd $(chromaphotonlist-dir); }
chromaphotonlist-scd(){  cd $(chromaphotonlist-sdir); }
chromaphotonlist-bcd(){  cd $(chromaphotonlist-bdir); }

chromaphotonlist-verbose(){ echo 1 ; }
chromaphotonlist-prefix(){ echo $(chromaphotonlist-dir) ; }

chromaphotonlist-geant4-home(){ 
  case $NODE_TAG in 
    D) echo /usr/local/env/chroma_env/src/geant4.9.5.p01 ;;
  esac
}
chromaphotonlist-geant4-dir(){ 
  case $NODE_TAG in 
    D) echo /usr/local/env/chroma_env/lib/Geant4-9.5.1 ;;
  esac
}
chromaphotonlist-rootsys(){
  case $NODE_TAG in 
    D) echo /usr/local/env/chroma_env/src/root-v5.34.14 ;;
  esac
}

chromaphotonlist-lib(){ echo $(chromaphotonlist-prefix)/lib/libChromaPhotonList.dylib ; }

chromaphotonlist-env(){      
   elocal- 
   export GEANT4_HOME=$(chromaphotonlist-geant4-home)
   export ROOTSYS=$(chromaphotonlist-rootsys)   # needed to find rootcint for dictionary creation
}

chromaphotonlist-export(){
   export CHROMAPHOTONLIST_LIB=$(chromaphotonlist-lib)
   export CHROMAPHOTONLIST_PREFIX=$(chromaphotonlist-dir)
}


chromaphotonlist-wipe(){
   local msg="=== $FUNCNAME :"
   local bdir="$(chromaphotonlist-bdir)"
   echo $msg deleting bdir $bdir
   rm -rf "$bdir"
}
chromaphotonlist-cmake(){
   type $FUNCNAME
   mkdir -p $(chromaphotonlist-bdir)   
   chromaphotonlist-bcd
   cmake -DGeant4_DIR=$(chromaphotonlist-geant4-dir) \
         -DCMAKE_INSTALL_PREFIX=$(chromaphotonlist-prefix) \
         $(chromaphotonlist-sdir) 

}
chromaphotonlist-make(){
   chromaphotonlist-bcd
   make $* VERBOSE=$(chromaphotonlist-verbose) 
}
chromaphotonlist-install(){
   chromaphotonlist-make install
}
chromaphotonlist-build(){
   chromaphotonlist-cmake
   chromaphotonlist-make
   chromaphotonlist-install
}
chromaphotonlist-build-full(){
   chromaphotonlist-wipe
   chromaphotonlist-build
}









