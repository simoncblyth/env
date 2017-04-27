# === func-gen- : graphics/yoctogl/yoctogl fgp graphics/yoctogl/yoctogl.bash fgn yoctogl fgh graphics/yoctogl
yoctogl-src(){      echo graphics/yoctogl/yoctogl.bash ; }
yoctogl-source(){   echo ${BASH_SOURCE:-$(env-home)/$(yoctogl-src)} ; }
yoctogl-vi(){       vi $(yoctogl-source) ; }
yoctogl-env(){      elocal- ; }
yoctogl-usage(){ cat << EOU

Yocto-gl
===========

Yocto/GL is a collection of single-file libraries for building physically-based
graphics applications. Yocto/GL is written in C++ and can be used from with C
or C++ and works on OSX (clang), Linux (clang/gcc) and Windows (cl).

Yocto/GL libraries are released under the permissive MIT license, while the
example apps are released under the 2-clause BSD (to include warranty for
binary distribution).

Discovered from list of C++ gltf viewers. 

* https://github.com/KhronosGroup/glTF#c
* https://github.com/xelatihy/yocto-gl
* https://libraries.io/github/xelatihy/yocto-gl

* http://pellacini.di.uniroma1.it

FABIO PELLACINI
ASSOCIATE PROFESSOR OF COMPUTER SCIENCE
SAPIENZA UNIVERSITY OF ROME


EOU
}
yoctogl-dir(){ echo $(local-base)/env/graphics/yoctogl/graphics/yoctogl-yoctogl ; }
yoctogl-cd(){  cd $(yoctogl-dir); }
yoctogl-mate(){ mate $(yoctogl-dir) ; }
yoctogl-get(){
   local dir=$(dirname $(yoctogl-dir)) &&  mkdir -p $dir && cd $dir

}
