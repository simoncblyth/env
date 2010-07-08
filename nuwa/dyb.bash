# === func-gen- : nuwa/dyb fgp nuwa/dyb.bash fgn dyb fgh nuwa
dyb-src(){      echo nuwa/dyb.bash ; }
dyb-source(){   echo ${BASH_SOURCE:-$(env-home)/$(dyb-src)} ; }
dyb-vi(){       vi $(dyb-source) ; }
dyb-env(){      
   elocal- ; 
   export DYB_RELEASE=$DYB/NuWa-trunk
   source $DYB_RELEASE/dybgaudi/Utilities/Shell/bash/dyb.sh
}
dyb-usage(){
  cat << EOU
     dyb-src : $(dyb-src)
     dyb-dir : $(dyb-dir)


EOU
}
dyb-dir(){ echo $(local-base)/env/nuwa/nuwa-dyb ; }
dyb-cd(){  cd $(dyb-dir); }
dyb-mate(){ mate $(dyb-dir) ; }
dyb-get(){
   local dir=$(dirname $(dyb-dir)) &&  mkdir -p $dir && cd $dir

}
dyb-rc(){ echo $HOME/.dybrc ; }

dyb-info(){
  env | grep DYB

  echo $(dyb-rc) ...
  cat $(dyb-rc)

}

dyb-init-(){  cat << EOS
do_setup=yes 
EOS
}

dyb-init(){
  $FUNCNAME- > $(dyb-rc)
}

dyb-edit(){
  vi $(dyb-rc)
}

