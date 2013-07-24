# === func-gen- : muon_simulation/weighted_optical_photon/opw fgp muon_simulation/weighted_optical_photon/opw.bash fgn opw fgh muon_simulation/weighted_optical_photon
opw-src(){      echo muon_simulation/optical_photon_weighting/opw.bash ; }
opw-source(){   echo ${BASH_SOURCE:-$(env-home)/$(opw-src)} ; }
opw-vi(){       vi $(opw-source) ; }
opw-env(){      elocal- ; }
opw-usage(){ cat << EOU

OPW
====

FUNCTIONS
---------

*opw-get*
    checkout my people copy of Davids OPW



EOU
}
opw-dir(){ echo $(local-base)/env/muon_simulation/optical_photon_weighting/OPW ; }
opw-cd(){  cd $(opw-dir); }
opw-mate(){ mate $(opw-dir) ; }
opw-get(){
   local dir=$(dirname $(opw-dir)) &&  mkdir -p $dir && cd $dir
   local nam=$(basename $(opw-dir))

   [ -z "$DYBSVN" ] && echo $msg missing DYBSVN && return 1 
   [ ! -d "$nam" ] && svn co $DYBSVN/people/blyth/$nam
   [   -d "$nam" ] && svn up $nam

}
