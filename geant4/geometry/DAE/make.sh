#!/bin/bash -l

main(){
  local arg=$1
  dae-
  if [ "$arg" == "clean" ]; then 
     dae-make clean
     dae-make && dae-install
  else
     dae-make && dae-install
  fi
}

export DYB=x
echo NB NON-STANDARD DYB $DYB
main $*


