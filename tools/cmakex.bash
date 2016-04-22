# === func-gen- : tools/cmakex fgp tools/cmakex.bash fgn cmakex fgh tools
cmakex-src(){      echo tools/cmakex.bash ; }
cmakex-source(){   echo ${BASH_SOURCE:-$(env-home)/$(cmakex-src)} ; }
cmakex-vi(){       vi $(cmakex-source) ; }
cmakex-env(){      elocal- ; }
cmakex-usage(){ cat << EOU

CMAKE Examples
================

staticlibs-add_subdir
-----------------------

* note linking with bare target name from another subdir 
  the library and needed dependent build is done automatically

* individual subdir build fails, but can build subdir libs
  from top level::

    simon:build blyth$ make help
    The following are some of the valid targets for this Makefile:
    ... all (the default if no target is provided)
    ... clean
    ... depend
    ... edit_cache
    ... rebuild_cache
    ... finally
    ... a
    ... b
    ... c
    ... main.o
    ... main.i
    ... main.s



EOU
}
cmakex-dir(){ echo $(local-base)/env/tools/cmake-examples ; }
cmakex-cd(){  cd $(cmakex-dir); }
cmakex-get(){
   local dir=$(dirname $(cmakex-dir)) &&  mkdir -p $dir && cd $dir

   git clone https://github.com/toomuchatonce/cmake-examples
}
