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
csa-nuwapkg-cd(){ cd $(csa-nuwapkg)/$1 ; }
csa-nuwapkg-cpto(){ 
   local iwd=$PWD 
   local pkg=$(csa-nuwapkg)
   local nam=DsChromaStackAction

   csa-cd

   cp src/$nam.h  $pkg/src/
   cp src/$nam.cc $pkg/src/

   perl -pi -e 's,ChromaPhotonList.hh,Chroma/ChromaPhotonList.hh,' $pkg/src/$nam.cc
   perl -pi -e 's,ZMQRoot.hh,ZMQRoot/ZMQRoot.hh,'                  $pkg/src/$nam.cc

   cd $iwd
}   

csa-nuwacfg(){
   local msg="=== $FUNCNAME :"
   local pkg=$1
   [ ! -d "$pkg/cmt" ] && echo ERROR NO cmt SUBDIR && sleep 1000000
   local iwd=$PWD

   echo $msg for pkg $pkg
   cd $pkg/cmt

   cmt config
   . setup.sh 

   cd $iwd
}

csa-nuwaenv(){

   zmqroot-
   csa-nuwacfg $(zmqroot-nuwapkg)

   cpl- 
   csa-nuwacfg $(cpl-nuwapkg)

   csa-
   csa-nuwacfg $(csa-nuwapkg)

}

csa-nuwarun(){

   #nuwa.py -n 1 -m "fmcpmuon --chroma --machinerytest"
   nuwa.py -n 1 -m "fmcpmuon --chroma"

}

