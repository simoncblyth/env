#!/bin/bash -l

main(){
   local name=${1:-mocksim}
   local msg="$name : " 
   local fenv=$HOME/env-dybx.sh
   [ -f "$fenv" ] && source $fenv && echo $msg sourced $fenv
   [ ! -f "$fenv" ] && echo $msg failed to find $fenv using default environment 

   export-
   export-export  # envvars pointing to geometry files

   $(local-base)/env/geant4/mocksim/$name $*
}


name=$(basename $0)
main ${name/.sh} 



