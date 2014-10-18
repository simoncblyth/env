# === func-gen- : root/rootsys fgp root/rootsys.bash fgn rootsys fgh root
rootsys-src(){      echo root/rootsys.bash ; }
rootsys-source(){   echo ${BASH_SOURCE:-$(env-home)/$(rootsys-src)} ; }
rootsys-vi(){       vi $(rootsys-source) ; }
rootsys-env(){      
    elocal- 
    rootsys-export
}
rootsys-usage(){ cat << EOU

ROOTSYS
=======

Used to define the ROOTSYS envvar which 
dictates which ROOT to use. This envvar is 
needed by cmake/Modules/FindROOT.cmake

FUNCTIONS
----------

*rootsys-export*


EOU
}

rootsys-rootsys(){
  case $NODE_TAG in 
     D) echo /usr/local/env/chroma_env/src/root-v5.34.14 ;;
  esac 
}
rootsys-export(){
   export ROOTSYS=$(rootsys-rootsys)
}



