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




echo
echo  ............ SEED ENV ..............
echo
env
echo

build
nuwa
setup

echo
echo ............ DERIVED ENV ... relativised wrt NUWA_HOME and PWD for presentation ............
echo
env |  perl -p -e "s,$NUWA_HOME/,,g" - | perl -p -e "s,$PWD,,g" - | sort   
echo


echo
echo ............ non- NUWA_HOME or PWD relative paths in derived env ............
echo
env |  perl -p -e "s,$NUWA_HOME/,,g" - | perl -p -e "s,$PWD,,g" - | sort | grep "=/"  
echo
echo



nosetests -v 

