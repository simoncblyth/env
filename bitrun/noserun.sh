#!/bin/sh

noserun_usage(){

cat << EOU

    Designed to be invoked from a bitten recipe with :
    
         bash /path/to/noserun.sh $path $config $build $revision 

    For test usage
        . ~/env/bitrun/noserun.sh

    noserun_     <home> <path> <config> <build> <revision> 
    noserun_cmt  <home> <path> <config> <build> <revision>

    <home> is set by virtue of the fixed depth of this script within the repository
    the others are passed as args, only path is needed when doing local tests           

    noserun_cmt $DDR dybgaudi/trunk/Simulation/GenTools

EOU

}

noserun_(){

  local msg="<!-- $FUNCNAME :"
  local esg=" -->"

  local home=$1 
  shift
  local branch=trunk


  ## these four come down from the master    
  local path=$1
  local config=$2
  local build=$3
  local revision=$4

  echo $msg path $path config ${config} build ${build} revision ${revision}  nargs $# $esg
  [ $# != 4 ] && echo $msg ERROR wrong number of arguments $esg && return 3 
  
  ## assuming working copy is a checkout of a single branch, usually "trunk" 
  ##  this does nothing to unittest/demo but removes trunk from trunk/unittest/demo 
  local strip=${path/$branch\//}
  
  ## absolute path to test base, in which tests are looked for
  local runhome=/Users/blyth/workflow
  local basepath=$runhome/$strip
  echo $msg home $home runhome $runhome branch $branch strip $strip basepath $basepath $esg  
  
  ## in real usage will pluck "nosetests" from path of CMT managed env, the xmlplug having been installed into
  ## NOSE_HOME
  ##
  local xmlplug=$home/unittest/nose/xmlplug.py  
  
  [ ! -f $xmlplug ]  && echo $msg ERROR no xmlplug $xmlplug   $esg && return 1
  [ ! -d $basepath ] && echo $msg ERROR no basepath $basepath $esg && return  2
  
  
  ## hmm the echos here go to stdout but nose is sending its output to stderr
  
  local cmd="python $xmlplug $basepath --with-xml-output --xml-format=bitten --xml-basepath=$basepath/ --xml-baseprefix= "
  
  # this causes problems of invalid token, see 
  #   http://dayabay.phys.ntu.edu.tw/tracs/env/ticket/30
  #echo $msg $cmd $esg
  eval $cmd

}



noserun_cmt(){

  local msg="<!-- $FUNCNAME :"
  local esg=" -->"

  local home=$1
  local branch=trunk
  shift 

  ## these four come down from the master  but for local testing just need path 
   
  local path=$1
  local config=$2
  local build=$3
  local revision=$4

  ## remove the positionals to void confusing CMT 
  set --

  ## assuming working copy is a checkout of a single branch, usually "trunk" 
  ##  this does nothing to unittest/demo but removes trunk from trunk/unittest/demo 
  local strip=${path/$branch\//}
  
  ## absolute path to test base, in which tests are looked for
  local basepath=$home/$strip

  echo $msg home $home branch $branch strip $strip basepath $basepath $esg  

  local setup=$home/setup.sh
  [ ! -f $setup ] && echo $msg ERROR no setup $setup   $esg && return 1

  ## avoid CMT warnings by starting clean 
  CMTEXTRATAGS=
  CMTPATH=
   
  . $home/setup.sh 

  [ ! -d $basepath ] && echo $msg ERROR no basepath $basepath $esg && return  2
  
  cd $basepath
  cd cmt
  
  [ ! -f setup.sh ] && cmt config
  . setup.sh
  
  cd $basepath
  
  local nosetests=$(which nosetests)
  
  if [ "$nosetests" == "" ]; then 
     cat << EOC
$msg 
   ERROR nosetests not available, 
   change/add private block of $basepath/cmt/requirements to : 
   
   private
   use DybTestPolicy
   end_private
      
$esg 
EOC

     return 3
  fi
  
  [ ! -x $nosetests ] && echo $msg ERROR nosetests $nosetests not found $esg && return 4
  
  # when the xmlplug is installed into NOSE_HOME
  #   nosetests  --with-xml-output --xml-format=bitten --xml-basepath=$basepath/ --xml-baseprefix= 
  #
  # this does not really belong in cmt fragments, given my move to minimizing CMT usage
  #  local xmlplug=$home/../installation/trunk/dybtest/scripts/xmlplug.py 
  #
   local xmlplug=$ENV_HOME/unittest/nose/xmlplug.py 
   python $xmlplug --with-xml-output --xml-format=bitten --xml-basepath=$basepath/ --xml-baseprefix=
 
}



## assumes depth of this script in the repository 
noserun_     $(dirname $(dirname $0)) $*
#noserun_cmt  $(dirname $(dirname $0)) $*
