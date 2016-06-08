optickswin-src(){      echo optickswin.bash ; }
optickswin-source(){   echo ${BASH_SOURCE:-$(env-home)/$(optickswin-src)} ; }
optickswin-vi(){       vi $(optickswin-source) ; }
optickswin-env(){      elocal- ; }
optickswin-usage(){ cat << EOU

Opticks Windows Port Notes
============================


Windows 7 MSYS2 build
-----------------------

Steps::
  
   # installed MSYS2 following web instructions to setup/update pacman

   pacman -S git
   pacman -S mercurial

   hg clone http://bitbucket.org/simoncblyth/env

   # hookup env to .bash_profile 
   # alias vi="vim"

   pacman -S cmake

   opticks-;opticks-externals-install   

   # glm- misses unzip
   pacman -S unzip

   # imgui- misses diff
   pacman -Ss diff
   pacman -S diffutils

   # assimp-get : connection reset by peer, but switch from git: to http: protocol and it works
   # openmesh-get : misses tar

   pacman -S tar 

   # glfw-cmake : runs into lack of toolchain

   pacman -S mingw-w64-x86_64-cmake    # from my notes msys2-

   # glfw-cmake : misses make

   pacman -S mingw-w64-x86_64-make     # guess
   pacman -S mingw-w64-x86_64-toolchain     # guess
   pacman -S base-devel           # guess
  
   # how to setup path to use the right ones ? 
   # see .bash_profile putting /mingw64/bin at head

   # adjust glfw-cmake imgui-cmake to specify opticks-cmake-generator 
   # glfw--   succeeds to cmake/make
   # glew--   succeeds to make (no cmake)
   # imgui-cmake   problems finding GLEW   
   #     maybe windows directory issue 
   #        what in MSYS2 appears as /usr/local/opticks/externals/
   #        is actually at C:\msys64\usr\local\opticks\externals\ 
   #     nope looks like windows lib naming, on mac its libGLEW.dylib on windows libglew32.a and libglew32.dll.a
   #     problem seems to be glew-get use of symbolic link, change glew-idir to avoid enables the find
   # imgui-cmake succeed
   # imgui-make  : undefined references in link to glGetIntegerv etc...
   #     
   # imgui-- : needed to configure the opengl libs and set defintion to avoid IME 
   #
   # assimp-cmake : fails to find DirectX : avoid by switch off tool building
   # opticks-cmake : fails to find Boost 

   pacman -S mingw-w64-x86_64-boost


   # for testing, need numpy and ipython
   pacman -S mingw-w64-x86_64-python2-numpy
   pacman -S mingw-w64-x86_64-python2-ipython 


Issues
--------

gdb : no stack
~~~~~~~~~~~~~~~~

Asserts are not trapped, so end up with no stack.

* http://stackoverflow.com/questions/2705442/debugging-mingw-program-with-gdb-on-windows-not-terminating-at-assert-failure

Try breakpointing exit, add to .gdbinit file with the lines:

   set breakpoint pending on
   b exit




EOU
}
optickswin-dir(){ echo $(opticks-home) ; }
optickswin-cd(){  cd $(optickswin-dir); }
