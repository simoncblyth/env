# === func-gen- : graphics/isosurface/implicitmesher fgp graphics/isosurface/implicitmesher.bash fgn implicitmesher fgh graphics/isosurface
implicitmesher-src(){      echo graphics/isosurface/implicitmesher.bash ; }
implicitmesher-source(){   echo ${BASH_SOURCE:-$(env-home)/$(implicitmesher-src)} ; }
implicitmesher-vi(){       vi $(implicitmesher-source) ; }
implicitmesher-env(){      elocal- ; }
implicitmesher-usage(){ cat << EOU


http://www.dgp.toronto.edu/~rms/software/ImplicitMesher/index.html

* ~/opticks_refs/Implicit_Surface_Polygonalizer_Bloomenthal.pdf



EOU
}
implicitmesher-edir(){ echo $(env-home)/graphics/isosurface/ImplicitMesher; }
implicitmesher-dir(){ echo $(local-base)/env/graphics/isosurface/ImplicitMesher; }
implicitmesher-cd(){  cd $(implicitmesher-dir); }

implicitmesher-get-original(){
   local dir=$(dirname $(implicitmesher-dir)) &&  mkdir -p $dir && cd $dir

   local url=http://www.dgp.toronto.edu/~rms/software/ImplicitMesher/ImplicitMesher.zip 
   local zip=$(basename $url)
   local nam=${zip/.zip}

   [ ! -f "$zip" ] && curl -L -O $url 
   [ ! -d "$nam" ] && unzip $zip

}

implicitmesher-ecd()
{
    local edir=$(implicitmesher-edir)
    mkdir -p $edir
    cd $edir
}

implicitmesher-make-original()
{
   glm-
   clang SimpleMesh.cpp ImplicitPolygonizer.cpp ImplicitFunction.cpp Blob.cpp Sphere.cpp main.cc -I$(glm-dir) -lc++ -o ImpM
}



implicitmesher-sdir(){   echo $HOME/ImplicitMesher ; }
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

    cmake \
       -DCMAKE_MODULE_PATH=$(opticks-home)/cmake/Modules \
       -DCMAKE_BUILD_TYPE=Debug \
       -DCMAKE_INSTALL_PREFIX=$(implicitmesher-prefix) \
       $* \
       $(implicitmesher-sdir)

}

implicitmesher--()
{
   local msg="$FUNCNAME : "
   local iwd=$PWD
   implicitmesher-bcd   

   cmake --build . --target ${1:-install}

   cd $iwd
}

implicitmesher-t()
{
   $(implicitmesher-prefix)/lib/ImplicitMesherTest $*
}

