#!/bin/bash -l

deploy-usage(){ cat << EOU

Workaround for not wanting to create a geant4 NuWa patch yet::

    DEPLOY_MODE="diff" ./deploy.sh 
    DEPLOY_MODE="ls -l" ./deploy.sh 
    DEPLOY_MODE="cp" ./deploy.sh 

EOU
}


deploy-paths-(){ cat << EOP
visualization/VRML/src/G4VRML2FileSceneHandler.cc
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
   g4-
}

deploy-mode(){ echo ${DEPLOY_MODE:-diff} ; }
deploy-source(){ echo $ENV_HOME/geant4/source ; }
deploy-target(){ echo $(g4-dir)/source ; }

deploy-main(){
   deploy-env
   deploy-paths 
}

deploy-main
deploy-usage

