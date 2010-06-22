#!/bin/bash
#
#   This script is used by bitten-slave recipe controlled runs ...
#
#   It must be invoked from a NuWa-<release> dir with for example : 
#       env -i BUILD_PATH=dybgaudi/trunk/RootIO/RootIOTest path/to/runtest.sh
# 
#
#   This was developed with the  "slv-" functions which provide simple 
#   recipe generation and testing of the recipes and this script.
#

rt-envdump(){
   echo
   echo
   echo  .....[$PWD]....... $* ..............
   case $1 in 
     derived) env |  perl -p -e "s,$NUWA_HOME/,,g" - | perl -p -e "s,$PWD,,g" - | sort   ;;
    absolute) env |  perl -p -e "s,$NUWA_HOME/,,g" - | perl -p -e "s,$PWD,,g" - | sort | grep "=/"  ;;
           *) env | sort ;;
   esac
   echo
   echo
}

## check running from required directory and with the mandatory seed environment
rt-assert(){
  local msg="=== $FUNCNAME :"
  [ -z "$BUILD_PATH" ]       && echo $msg ERROR BUILD_PATH is not defined && return 1
  return 0
}

## release from dirname eg NuWa-trunk => trunk
rt-release(){   
  local nw="NuWa-"
  [ "${1:0:${#nw}}" == "$nw" ] && echo ${1:${#nw}} || echo $1
}

## working copy dir from repository path  eg dybgaudi/trunk/RootIO/RootIOTest => dybgaudi/RootIO/RootIOTest
rt-wcdir(){     
   local elems=$(echo $1 | tr "/" " ")
   local dir=""
   local elem
   for elem in $elems ; do
      [ "$elem" != "$NUWA_RELEASE" ] && dir="$dir $elem" 
   done
   echo $dir | tr " " "/"
}

## first path relative to the second 
rt-relativeto(){ [ "$1" == "$2" -o "$2" == "" ] && echo "" || echo ${1/$2\//} ; }

## CAUTION : xmlout is constrained to match the python:unittest attribute in the recipe
rt-testname(){  echo $(basename $1) | tr "[A-Z]" "[a-z]" ; }
rt-xmlout(){    echo $NUWA_HOME/../test-$(rt-testname $1).xml ;  }


## get into environment and directory for running test 
rt-envsetup(){
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

rt-main(){

   ## invokation directory is regarded as NUWA_HOME ... it should have name of form "NuWa-trunk"
   rt-envdump seed
   rt-assert
   [ "$?" != "0" ] && echo rt-assert failure exiting && exit 1 

   export NUWA_HOME=$PWD
   export NUWA_RELEASE=$(rt-release $(basename $NUWA_HOME))
   export NUWA_TESTDIR=$(rt-wcdir $BUILD_PATH)

   export BUILD_MASTERPATH=${BUILD_MASTERPATH:-/}
   export BUILD_BASEPREFIX=$(rt-relativeto $BUILD_PATH $BUILD_MASTERPATH)/ 
   export BUILD_XMLOUT=$(rt-xmlout $BUILD_PATH)

   rt-envdump inferred 

   rt-envsetup

   rt-envdump derived
   rt-envdump absolute

   local cmd="nosetests --with-xml-output --xml-outfile=$BUILD_XMLOUT --xml-baseprefix=$BUILD_BASEPREFIX"
   echo $msg $cmd   
   eval $cmd
}


rt-main
