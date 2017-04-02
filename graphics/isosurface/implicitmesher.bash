# === func-gen- : graphics/isosurface/implicitmesher fgp graphics/isosurface/implicitmesher.bash fgn implicitmesher fgh graphics/isosurface
implicitmesher-src(){      echo graphics/isosurface/implicitmesher.bash ; }
implicitmesher-source(){   echo ${BASH_SOURCE:-$(env-home)/$(implicitmesher-src)} ; }
implicitmesher-vi(){       vi $(implicitmesher-source) ; }
implicitmesher-usage(){ cat << EOU

Implicit Mesher
=================

* https://bitbucket.org/simoncblyth/implicitmesher

* http://www.dgp.toronto.edu/~rms/software/ImplicitMesher/index.html

* ~/opticks_refs/Implicit_Surface_Polygonalizer_Bloomenthal.pdf

Dependencies
--------------

std::function from functional header

     * requires C++11 ?
     * could perhaps rejig to use boost::function to support older compilers

glm headers
    


EOU
}

implicitmesher-dir-original(){ echo $(local-base)/env/graphics/isosurface/ImplicitMesher; }
implicitmesher-cd-original(){  cd $(implicitmesher-dir-original); }
implicitmesher-get-original(){
   local dir=$(dirname $(implicitmesher-dir-original)) &&  mkdir -p $dir && cd $dir

   local url=http://www.dgp.toronto.edu/~rms/software/ImplicitMesher/ImplicitMesher.zip 
   local zip=$(basename $url)
   local nam=${zip/.zip}

   [ ! -f "$zip" ] && curl -L -O $url 
   [ ! -d "$nam" ] && unzip $zip

}


implicitmesher-make-original()
{
   glm-
   clang SimpleMesh.cpp ImplicitPolygonizer.cpp ImplicitFunction.cpp Blob.cpp Sphere.cpp main.cc -I$(glm-dir) -lc++ -o ImpM
}

implicitmesher-env(){      elocal- ; implicitmesher-export ; }


implicitmesher-dir(){    echo $HOME/ImplicitMesher ; }
implicitmesher-sdir(){   echo $HOME/ImplicitMesher ; }
implicitmesher-cd(){    cd $(implicitmesher-dir)/$1 ; }

implicitmesher-bdir(){   echo $LOCAL_BASE/env/graphics/ImplicitMesher.build ; }
implicitmesher-prefix(){ echo $LOCAL_BASE/env/graphics/ImplicitMesher ; }
implicitmesher-bcd(){    cd $(implicitmesher-bdir) ; }

implicitmesher-edit(){ vi $(implicitmesher-sdir)/CMakeLists.txt ; }

implicitmesher-cmake()
{
    local bdir=$(implicitmesher-bdir)

    mkdir -p $bdir
    #[ -f "$bdir/CMakeCache.txt" ] && echo $msg configured already && return  
    rm -f "$bdir/CMakeCache.txt"

    implicitmesher-bcd   
    opticks-
    # notice no glm precursor here, it comes it a CMake level 

    cmake \
       -DCMAKE_MODULE_PATH=$(opticks-home)/cmake/Modules \
       -DCMAKE_BUILD_TYPE=Debug \
       -DCMAKE_INSTALL_PREFIX=$(implicitmesher-prefix) \
       $* \
       $(implicitmesher-sdir)

}

implicitmesher-export()
{
   local libdir=$(implicitmesher-prefix)/lib
   [ "${PATH/$libdir}" == "$PATH"  ] && export PATH=$libdir:$PATH || echo $msg libdir $libdir already in PATH
}


implicitmesher--()
{
   local msg="$FUNCNAME : "
   local iwd=$PWD
   implicitmesher-bcd   

   cmake --build . --target ${1:-install}

   cd $iwd
}


implicitmesher-bin(){ echo $(implicitmesher-prefix)/lib/ImplicitMesherTest ; }
implicitmesher-t(){  $(implicitmesher-bin) $* ;}
implicitmesher-d(){  lldb $(implicitmesher-bin) $* ;}


implicitmesher-standalone()
{
   implicitmesher-cd tests

    glm-
   #clang ImplicitMesherFTest.cc -I$(implicitmesher-prefix)/include/ImplicitMesher -I$(implicitmesher-dir) -I$(glm-dir) -c -o /dev/null


   local incs="-I$(implicitmesher-dir) -I$(glm-dir) "
   local libs="-L$(implicitmesher-prefix)/lib -lImplicitMesher"

   clang ImplicitMesherFTest.cc $incs $libs -std=c++11 -stdlib=libc++ -lc++ -o /tmp/ImplicitMesherFTest

   DYLD_LIBRARY_PATH=$(implicitmesher-prefix)/lib lldb /tmp/ImplicitMesherFTest

}



