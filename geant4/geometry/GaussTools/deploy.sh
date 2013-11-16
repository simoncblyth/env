#!/bin/bash -l

deploy-usage(){ cat << EOU

Workaround not wanting to commit to dybsvn yet, as
liable not to compile with vanilla NuWa install.

::

    DEPLOY_MODE="diff" ./deploy.sh 
    DEPLOY_MODE="ls -l" ./deploy.sh 
    DEPLOY_MODE="cp" ./deploy.sh 

EOU
}


deploy-paths-(){ cat << EOP
cmt/requirements
src/Components/GiGaRunActionGDML.cpp
src/Components/GiGaRunActionGDML.h
EOP
}

deploy-paths(){
  local msg="=== $FUNCNAME :"
  local source=$(deploy-source)
  local target=$(deploy-target)
  local mode=$(deploy-mode)
  local path
  local cmd
  $FUNCNAME- | while read path ; do 
     cmd="$mode $source/$path $target/$path"
     echo $msg $cmd 
     eval $cmd
  done
}

deploy-env(){
   nuwa-
}

deploy-mode(){ echo ${DEPLOY_MODE:-diff} ; }
deploy-source(){ echo $ENV_HOME/geant4/geometry/GaussTools ; }
deploy-target(){ echo $(nuwa-lhcb-dir)/Sim/GaussTools ; }

deploy-main(){
   deploy-env
   deploy-paths 
   #deploy-build
}

deploy-build(){
   cd $(deploy-target)/cmt
   fenv
   cmt config 
   . setup.sh
   make
}

deploy-main
deploy-usage

