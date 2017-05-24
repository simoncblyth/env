# === func-gen- : graphics/yoctogl/yoctogl fgp graphics/yoctogl/yoctogl.bash fgn yoctogl fgh graphics/yoctogl
yoctogl-src(){      echo graphics/yoctogl/yoctogl.bash ; }
yoctogl-source(){   echo ${BASH_SOURCE:-$(env-home)/$(yoctogl-src)} ; }
yoctogl-vi(){       vi $(yoctogl-source) ; }
yoctogl-usage(){ cat << EOU

Yocto-gl
===========


DONT USE THIS CLONE : USE THE oyoctogl ONE 
------------------------------------------


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


Compilation Error
-------------------

::

    delta:yoctogl-test-dir.build blyth$ yoctogl-;yoctogl-test-make
    In file included from /Users/blyth/env/graphics/yoctogl/ygltf_reader.cc:5:
    /usr/local/env/graphics/yoctogl/yocto-gl/yocto/yocto_gltf.h:231:30: error: chosen constructor is explicit in copy-initialization
        extension_t extensions = {};
                                 ^~
    /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/../include/c++/v1/map:838:14: note: constructor declared here
        explicit map(const key_compare& __comp = key_compare())
                 ^

Avoided with::

    delta:yocto-gl blyth$ clang++ -v
    Apple LLVM version 6.0 (clang-600.0.57) (based on LLVM 3.5svn)
    Target: x86_64-apple-darwin13.3.0
    Thread model: posix

     struct glTFProperty_t {
         /// No description in schema.
    -    extension_t extensions = {};
    +    //extension_t extensions = {};
    +    extension_t extensions{};
         /// No description in schema.
         extras_t extras = {};
     };


EOU
}

yoctogl-edir(){   echo $(env-home)/graphics/yoctogl ; }
yoctogl-dir(){    echo $(local-base)/env/graphics/yoctogl/yocto-gl ; }
yoctogl-prefix(){ echo $(yoctogl-dir)/yocto ; }
yoctogl-cd(){   cd $(yoctogl-dir); }
yoctogl-ecd(){  cd $(yoctogl-edir); }
yoctogl-get(){
   local dir=$(dirname $(yoctogl-dir)) &&  mkdir -p $dir && cd $dir
   [ ! -d "yocto-gl" ] && git clone https://github.com/simoncblyth/yocto-gl
}

yoctogl-env(){      elocal- ; gltf- ;  echo DONT USE THIS USE oyoctogl- ; }
yoctogl-manual()
{
    yoctogl-ecd

    local dir=$TMP/$FUNCNAME
    mkdir -p $dir 

    clang++ -std=c++1y gltf_reader.cc -I$(yoctogl-prefix) -lc++ -o $dir/gltf_reader

    local minimal=$(gltf-minimal-sample)
    $dir/reader $minimal
}

yoctogl-find(){ 
    local iwd=$PWD
    yoctogl-cd
    find . -type f -exec grep -H ${1:-float4x4} {} \;
    #cd $iwd
}


yoctogl-test-dir(){  echo $TMP/$FUNCNAME ; }
yoctogl-test-bdir(){ echo $(yoctogl-test-dir).build ; }
yoctogl-test-bcd(){  local bdir=$(yoctogl-test-bdir) ; mkdir -p $bdir ; cd $bdir ; }
yoctogl-test-cmake()
{
    yoctogl-test-bcd
    cmake \
       -DCMAKE_BUILD_TYPE=Debug \
       -DCMAKE_INSTALL_PREFIX=$(yoctogl-test-dir) \
       $* \
       $(yoctogl-edir)
}

yoctogl-test-make()
{
    yoctogl-test-bcd
    make 
    make install
}

yoctogl-test()
{
    #local path=$(gltf-minimal-sample)
    local path=$TMP/nd/scene.gltf 

    $TMP/yoctogl-test-dir/bin/ygltf_reader $path
}



