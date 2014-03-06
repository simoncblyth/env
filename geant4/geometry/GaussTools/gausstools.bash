# === func-gen- : geant4/geometry/GaussTools/gausstools fgp geant4/geometry/GaussTools/gausstools.bash fgn gausstools fgh geant4/geometry/GaussTools
gausstools-src(){      echo geant4/geometry/GaussTools/gausstools.bash ; }
gausstools-source(){   echo ${BASH_SOURCE:-$(env-home)/$(gausstools-src)} ; }
gausstools-vi(){       vi $(gausstools-source) ; }
gausstools-usage(){ cat << EOU

GAUSSTOOLS
=============



EOU
}
gausstools-dir(){ echo $(env-home)/geant4/geometry/GaussTools ; }
gausstools-cd(){  cd $(gausstools-dir); }
gausstools-mate(){ mate $(gausstools-dir) ; }
gausstools-get(){
   local dir=$(dirname $(gausstools-dir)) &&  mkdir -p $dir && cd $dir

}
gausstools-env(){      elocal- ; nuwa- ; }
gausstools-home(){   echo $ENV_HOME/geant4/geometry/GaussTools ; }
gausstools-target(){ echo $(nuwa-lhcb-dir)/Sim/GaussTools ; }
gausstools-mode(){   echo ${GAUSSTOOLS_MODE:-diff} ; }
gausstools-paths-(){ cat << EOP
cmt/requirements
src/Components/GiGaRunActionGDML.cpp
src/Components/GiGaRunActionGDML.h
EOP
}

gausstools-deploy(){
  local msg="=== $FUNCNAME :"
  local home=$(gausstools-home)
  local target=$(gausstools-target)
  local mode=$(gausstools-mode)
  local path
  local cmd
  gausstools-paths- | while read path ; do 
     cmd="$mode $home/$path $target/$path"
     echo $msg $cmd 
     eval $cmd
  done
}

gausstools-make(){
   cd $(gausstools-target)/cmt
   fenv
   cmt config 
   . setup.sh
   make
}

gausstools-diff(){ GAUSSTOOLS_MODE="diff" gausstools-deploy ; }
gausstools-cp(){   GAUSSTOOLS_MODE="cp"   gausstools-deploy ; }

gausstools-update(){
   gausstools-diff
   gausstools-cp
   gausstools-make
}






