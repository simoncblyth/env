# === func-gen- : tools/tools fgp tools/tools.bash fgn tools fgh tools
tools-src(){      echo tools/tools.bash ; }
tools-source(){   echo ${BASH_SOURCE:-$(env-home)/$(tools-src)} ; }
tools-vi(){       vi $(tools-source) ; }
tools-env(){      elocal- ; }
tools-usage(){ cat << EOU

Env Tools
==========

.. toctree:: 

    fabric
    cuisine
    daemonwatch



EOU
}
tools-dir(){ echo $(local-base)/env/tools/tools-tools ; }
tools-cd(){  cd $(tools-dir); }
tools-mate(){ mate $(tools-dir) ; }
tools-get(){
   local dir=$(dirname $(tools-dir)) &&  mkdir -p $dir && cd $dir

}
