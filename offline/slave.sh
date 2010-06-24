#!/bin/bash
#  
#      Slave runner script, 
#
#          * contacts the master via http to see if there is a pending build to perform. 
#          * if so then gets the step by step recipe to perform the build from the master 
#            and follows the recipe reporting progress back to the master step by step.
#
#      Usage :
#                cd /path/to/dybinst/export
#                ./installation/trunk/dybinst/scripts/slave.sh <release> [<build_dir>]
#
#          OR equivalently invoked from dybinst
#    
#                ./dybinst trunk slave 
#                ./dybinst -b build_\\\${config}_\\\${revision} trunk slave 
#                ./dybinst -b \\\${slv.stamp}  trunk slave
#
#           NB backslashes for protecting the dollars from the shell
#
#
#      Arguments :
#            <release> : 
#
#                   typically trunk
#  
#            <build_dir> : 
#
#                   When not defined or defined as "." indicates an "update" build to be done here
#                   if specified indicates a "full" build to be done in the directory provided.
#
#                   The directory can be specified using a pattern that is evaluated
#                   in the context supplied by the master and also locally created by this script
#                   at runtime.
#
#                   Using context supplied by the master  : "build_\${config}_\${build}_\${revision}"
#
#                             "config" :  name identifying the recipe eg "dybinst"
#                              "build" :  build number 
#                           "revision" :  svn revision number
#
#                   Local context created by this script : "\${slv.stamp}"
#
#       The "slv." context is derived from variables starting "slv_" in $HOME/.dybinstrc
#       For example :
#
#             slv_username=slave
#             slv_password=youknowit
#             slv_buildsurl=http://dayabay.ihep.ac.cn/tracs/dybsvn/builds
#
#             slv_logurl=http://cms01.phys.ntu.edu.tw:8181/logs/dybinst
#             slv_stamp=NuWa-$(date +"%Y%m%d")
#
#      The first three are mandatory providing url of the master instance and credentials
#      with which to contact it. 
#   
#      For local testing of slave running replace the slv_buildsurl with the path to a recipe, eg 
#             slv_dyb=/usr/local/dyb
#             slv_buildsurl=$slv_dyb/dybinst.xml
#
#

slv-cfg-msg(){  cat << EOM
# DO NOT EDIT : derived by $0  $(date) 
EOM
}

slv-cfg-localctx(){  cat << EOL
# spoof "master" context locally for recipe running without the master
[local]
path = /
build = 1000
config = dybinst
revision = 8800
EOL
}

slv-cfg-nuwa(){ cat << EOC
[nuwa]
release = $1
EOC
}

slv-cfg-dybini(){
  cat << EOH
# slv_ prefixed config vars from translated for slave consumption
[slv]
EOH
  local key ; for key in ${!slv_*} ; do
     eval local val=\$$key
     echo ${key:4}=$val
  done 
}


slv-cfg(){  
  local src=$1
  local release=$2
  slv-cfg-msg
  slv-cfg-localctx
  slv-cfg-dybini $src
  slv-cfg-nuwa   $release
}

slv-cfg-path(){ echo $HOME/.bitten-slave/$1.cfg ; }


slv-main(){
   local msg="=== $FUNCNAME :"
   local release=$1     ; shift
   local build_dir=$1   ; shift
   local opts=$*
   local mode=$( [ -z "$build_dir" -o "$build_dir" == "." ] && echo update || echo full )

   local name=${mode}_$(hostname)
   local log=${name}.log  
   local cfg=$(slv-cfg-path $name)

   local src=$HOME/.dybinstrc
   [ -f $src ] && . $src       ## context pollution access 
   [ -z "$slv_username" ]  && echo $msg ERROR slv_username is not defined in $src && return 1
   [ -z "$slv_password" ]  && echo $msg ERROR slv_password is not defined in $src && return 1
   [ -z "$slv_buildsurl" ] && echo $msg ERROR slv_buildsurl is not defined in $src && return 1

   local url=$slv_buildsurl
   local xopt=$([ "${url:0:4}" == "http" ] && echo -n || echo "--dry-run" )    ## dry run local recipe running 

   mkdir -p $(dirname $cfg)
   
   slv-cfg $src $release > $cfg 
   chmod go-rw $cfg 
   echo $msg derive config $cfg from source $src 
   cat $cfg

   local cmd="bitten-slave $xopt $opts --work-dir=. --build-dir=${build_dir:-.} --keep-files --name=$name --config=$cfg  --log=$log --user=$slv_username --password=$slv_password $url "
   echo $msg $cmd
   eval $cmd
}


slv-main $*



