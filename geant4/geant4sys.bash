# === func-gen- : geant4/geant4sys fgp geant4/geant4sys.bash fgn geant4sys fgh geant4
geant4sys-src(){      echo geant4/geant4sys.bash ; }
geant4sys-source(){   echo ${BASH_SOURCE:-$(env-home)/$(geant4sys-src)} ; }
geant4sys-vi(){       vi $(geant4sys-source) ; }
geant4sys-env(){      
     elocal- 
     export GEANT4_HOME=$(geant4sys-home)

}
geant4sys-usage(){ cat << EOU

GEANT4SYS
==========

Picking which G4 to use on a node in analogy to rootsys-


EOU
}
geant4sys-dir(){ echo $(local-base)/env/geant4/geant4-geant4sys ; }
geant4sys-cd(){  cd $(geant4sys-dir); }

geant4sys-home(){
   case $NODE_TAG in 
      D) echo /usr/local/env/chroma_env/src/geant4.9.5.p01 ;;  # chroma-geant4-sdir  
      *) echo not-configured-see-geant4sys- ;; 
   esac
}



    


