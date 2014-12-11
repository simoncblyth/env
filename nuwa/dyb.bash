# === func-gen- : nuwa/dyb fgp nuwa/dyb.bash fgn dyb fgh nuwa
dyb-src(){      echo nuwa/dyb.bash ; }
dyb-source(){   echo ${BASH_SOURCE:-$(env-home)/$(dyb-src)} ; }
dyb-vi(){       vi $(dyb-source) ; }
dyb-env(){      
   elocal- ; 
   export DYB_RELEASE=$DYB/NuWa-trunk
   export DYBSVN=http://dayabay.ihep.ac.cn/svn/dybsvn 
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
  ls -l  $DYB_RELEASE/dybgaudi/Utilities/Shell/bash/dyb.sh
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

dyb-setup(){
  [ -z "$DYB"  ]               && echo $msg ERROR need DYB to point to dybinst export directory && return 1
  cd $DYB
  . NuWa-trunk/setup.sh 
}

dyb-d(){
  [ -z "$DYB"  ]               && echo $msg ERROR need DYB to point to dybinst export directory && return 1
  cd $DYB
}


dyb--(){
   local msg="=== $FUNCNAME :"
   [ -z "$DYB"  ]               && echo $msg ERROR need DYB to point to dybinst export directory && return 1
   [ -z "$DYB_RELEASE" ]        && export DYB_RELEASE=$DYB/NuWa-${DYB_VERSION:-trunk}            && env | grep DYB_
   ! type dyb 1>/dev/null 2>&1  && echo $msg defining dyb function                               && source $DYB_RELEASE/dybgaudi/Utilities/Shell/bash/dyb.sh
   [ ! -f "$HOME/.dybrc" ]      && echo $msg creating $HOME/.dybrc                               && echo do_setup=yes > $HOME/.dybrc
   [ -z "$DYBRELEASEROOT" ]     && echo $msg invoking \"dyb dybgaudi\"                           && dyb dybgaudi
   [ -n "$*" ]                  && echo $msg invoking \"dyb $*\"     && dyb $*
}


rootiotest(){         dyb-- $FUNCNAME ; } 
dybpython(){          dyb-- $FUNCNAME ; cd python/DybPython ;  }
dybdbi(){             dyb-- $FUNCNAME ; cd python/DybDbi ;  }





