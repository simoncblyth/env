#!/bin/bash -l

main(){
   local name=${1:-gdmltest}
   local msg="$name : " 
   local fenv=$HOME/env-dybx.sh
   [ -f "$fenv" ] && source $fenv && echo $msg sourced $fenv
   [ ! -f "$fenv" ] && echo $msg failed to find $fenv using default environment 

   export-
   export-export

   $(local-base)/env/geant4/geometry/gdml/$name $*
}


name=$(basename $0)
main ${name/.sh} 



