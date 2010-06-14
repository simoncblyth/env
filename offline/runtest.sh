#!/bin/bash
#
#  Intended invokation ...
#
#     cd /dir/with/dybinst
#     env -i NUWA_HOME=$PWD/NuWa-trunk NUWA_TESTDIR=dybgaudi/RootIO/RootIOTest path/to/runtest.sh
#
#

build(){
  export BUILD_PWD=$PWD 
  export BUILD_PATH=/ 
  export BUILD_CONFIG_PATH=/ 
  export BUILD_REVISION= 
  export BUILD_NUMBER=1000  
}

nuwa(){
  #export NUWA_HOME=$PWD/NuWa-trunk 
  export NUWA_LOGURL=dummy
  #export NUWA_TESTDIR=dybgaudi/RootIO/RootIOTest
}

setup(){
  unset SITEROOT 
  unset CMSPROJECTPATH 
  unset CMTPATH 
  unset CMTEXTRATAGS 
  unset CMTCONFIG 

  . $NUWA_HOME/setup.sh  

  cd $NUWA_HOME/dybgaudi/DybRelease/cmt 
  [ ! -f setup.sh ] && cmt config ; . setup.sh ; cd .. 

  cd $NUWA_HOME/$NUWA_TESTDIR/cmt 
  [ ! -f setup.sh ] && cmt config ; . setup.sh ; cd .. 
}


echo ############ SEED ENV ############
env
echo
echo 
echo


build
nuwa
setup

echo ############ DERIVED ENV ############
env


