#!/bin/bash

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
  local src=$1
  cat << EOH
# slv_ prefixed config vars from $src translated for slave consumption
[slv]
EOH
  [ -f "$src" ] && . $src
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

slv-cfg-path(){ echo $HOME/.bitten-slave/dybsvn.cfg ; }
slv-main(){
   local msg="=== $FUNCNAME :"
   local release=$1

   ## derive config from the src 
   local src=$HOME/.dybinstrc
   local cfg=$(slv-cfg-path)
   mkdir -p $(dirname $cfg)
   slv-cfg $src $release > $cfg 
   chmod go-rw $cfg 
   echo $msg translated $src into $cfg 
   cat $cfg


   local log=slv.log  
   local url=recipe.xml

   bitten-slave --dry-run --config=$cfg --work-dir=. --build-dir=. --verbose --keep-files --log=$log --user=\${slv.username} --password=\${slv.password} $url
}


slv-main $*



