# === func-gen- : nuwa/nenv fgp nuwa/nenv.bash fgn nenv fgh nuwa
nenv-src(){      echo nuwa/nenv.bash ; }
nenv-source(){   echo ${BASH_SOURCE:-$(env-home)/$(nenv-src)} ; }
nenv-vi(){       vi $(nenv-source) ; }
nenv-env(){      
   elocal- ; 
   nenv-source
}
nenv-usage(){
  cat << EOU
     nenv-src : $(nenv-src)
     nenv-dir : $(nenv-dir)

    https://wiki.bnl.gov/dayabay/index.php?title=Environment_Management_with_nuwaenv

     DYB : $DYB  envvar that points to the dybinst directory 
       
     nenv-install
         generates $(nenv-name).sh in $(nenv-dir)

     nenv-
         sources $(nenv-name).sh that defines the nuwaenv function 

     Check its defined ... 

         nuwaenv --help



EOU
}
nenv-dir(){ echo $(local-base)/env/nuwa ; }
nenv-cd(){  cd $(nenv-dir); }
nenv-mate(){ mate $(nenv-dir) ; }
nenv-get(){
   local dir=$(dirname $(nenv-dir)) &&  mkdir -p $dir && cd $dir
}

nenv-name(){ echo nuwaenv ; }
nenv-install(){
  mkdir -p $(nenv-dir)
  local cmd="$DYB/installation/trunk/dybinst/python/nuwaenv.py -S $(nenv-dir)/$(nenv-name) "
  echo $msg $cmd ...
  eval $cmd 
}
nenv-source(){
  source $(nenv-dir)/$(nenv-name).sh || echo you need to nenv-install 1st 
}

nenv-init(){
  $FUNCNAME- > $HOME/.nuwaenv.cfg
}

nenv-init-(){ cat << EOC
[defaults]
base_release = $DYB/NuWa-trunk

EOC
}


nenv-dyb-init(){
   export DYB_RELEASE=$DYB/NuWa-trunk
   source $DYB_RELEASE/dybgaudi/Utilities/Shell/bash/dyb.sh
}


