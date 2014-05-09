# === func-gen- : nuwa/detsim/csa fgp nuwa/detsim/csa.bash fgn csa fgh nuwa/detsim
csa-src(){      echo nuwa/detsim/csa.bash ; }
csa-source(){   echo ${BASH_SOURCE:-$(env-home)/$(csa-src)} ; }
csa-vi(){       vi $(csa-source) ; }
csa-env(){      elocal- ; }
csa-usage(){ cat << EOU





EOU
}
csa-dir(){ echo $(env-home)/nuwa/detsim ; }
csa-cd(){  cd $(csa-dir); }
csa-mate(){ mate $(csa-dir) ; }
csa-get(){
   local dir=$(dirname $(csa-dir)) &&  mkdir -p $dir && cd $dir

}

csa-nuwapkg(){ echo $DYB/NuWa-trunk/dybgaudi/Simulation/DetSimChroma ; }
csa-nuwapkg-cd(){ cd $(csa-nuwapkg) ; }
csa-nuwapkg-cpto(){ 
   local iwd=$PWD 
   local pkg=$(csa-nuwapkg)
   local nam=DsChromaStackAction

   csa-cd

   cp src/$nam.h  $pkg/src/
   cp src/$nam.cc $pkg/src/

   cd $iwd
}   


