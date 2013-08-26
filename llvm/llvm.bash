# === func-gen- : llvm/llvm fgp llvm/llvm.bash fgn llvm fgh llvm
llvm-src(){      echo llvm/llvm.bash ; }
llvm-source(){   echo ${BASH_SOURCE:-$(env-home)/$(llvm-src)} ; }
llvm-vi(){       vi $(llvm-source) ; }
llvm-env(){      elocal- ; }
llvm-usage(){ cat << EOU

LLVM
=====

* http://llvm.org/

The LLVM Project is a collection of modular and reusable compiler and toolchain
technologies. Despite its name, LLVM has little to do with traditional virtual
machines, though it does provide helpful libraries that can be used to build
them. The name "LLVM" itself is not an acronym; it is the full name of the
project.



* http://llvm.org/docs/GettingStarted.html#overview

Configure
----------

::

    checking for GCC atomic builtins... no
    configure: WARNING: LLVM will be built thread-unsafe because atomic builtins are missing



EOU
}
llvm-dir(){ echo $(local-base)/env/llvm/llvm ; }
llvm-cd(){  cd $(llvm-dir)/$1; }
llvm-mate(){ mate $(llvm-dir) ; }
llvm-get(){
   local dir=$(dirname $(llvm-dir)) &&  mkdir -p $dir && cd $dir
   local iwd=$dir

   [ ! -d llvm ] && svn co http://llvm.org/svn/llvm-project/llvm/trunk llvm 

   cd $iwd/llvm/tools
   [ ! -d clang ] && svn co http://llvm.org/svn/llvm-project/cfe/trunk clang

   cd $iwd/llvm/projects
   [ ! -d compiler-rt ] && svn co http://llvm.org/svn/llvm-project/compiler-rt/trunk compiler-rt

   cd $iwd/llvm/projects
   [ ! -d test-suite ] && svn co http://llvm.org/svn/llvm-project/test-suite/trunk test-suite

}


llvm-build(){

   llvm-cd ..
   mkdir -p build
   cd build

   ../llvm/configure  --with-python=/usr/bin/python26

   # maybe should build a newer gcc and point to it using --with-gcc-toolchain=

}
