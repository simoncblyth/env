# === func-gen- : tools/realpath/realpath fgp tools/realpath/realpath.bash fgn realpath fgh tools/realpath
realpath-src(){      echo tools/realpath/realpath.bash ; }
realpath-source(){   echo ${BASH_SOURCE:-$(env-home)/$(realpath-src)} ; }
realpath-vi(){       vi $(realpath-source) ; }
realpath-env(){      elocal- ; }
realpath-usage(){ cat << EOU





EOU
}
realpath-dir(){ echo $(env-home)/tools/realpath ; }
realpath-cd(){  cd $(realpath-dir); }

realpath--(){
   local msg="=== $FUNCNAME :"
   local bin=$HOME/env/bin/realpath

   [ -f "$bin" ] && echo $msg bin $bin exists already && return 

   clang $(realpath-dir)/realpath.c -o $bin
}

