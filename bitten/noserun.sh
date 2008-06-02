#!/bin/sh
#
#   invoke from bitten recipe with 
#       bash /path/to/noserun.sh $path $config $build $revision 
#   
#
#

noserun(){

  local msg="=== $FUNCNAME :"


  ## these four come down from the master    
  local path=$1
  local config=$2
  local build=$3
  local revision=$4

  local home=$5
  local branch=trunk
  
  echo $msg path $path config ${config} build ${build} revision ${revision}  nargs $# 
  [ $# != 5 ] && echo $msg ERROR wrong number of arguments && return 3 
  
  ## assuming working copy is a checkout of a single branch, usually "trunk" 
  ##  this does nothing to unittest/demo but removes trunk from trunk/unittest/demo 
  local strip=${path/$branch\//}
  
  ## absolute path to test base, in which tests are looked for
  local basepath=$home/$strip
  echo $msg home $home branch $branch strip $strip basepath $basepath
  
  ## in real usage will pluck "nosetests" from path of CMT managed env, the xmlplug having been installed into
  ## NOSE_HOME
  ##
  local xmlplug=$home/unittest/nose/xmlplug.py  
  
  [ ! -f $xmlplug ]  && echo $msg ERROR no xmlplug $xmlplug && return 1
  [ ! -d $basepath ] && echo $msg ERROR no basepath $basepath && return  2
  
  local cmd="python $xmlplug $basepath --with-xml-output --xml-format=bitten --xml-basepath=$basepath "
  echo $cmd
  eval $cmd

}


## assumes depth of this script in the repository 
noserun $* $(dirname $(dirname $0))
