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


