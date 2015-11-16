# === func-gen- : graphics/geometry/icosahedron/icosahedron fgp graphics/geometry/icosahedron/icosahedron.bash fgn icosahedron fgh graphics/geometry/icosahedron
icosahedron-src(){      echo graphics/geometry/icosahedron/icosahedron.bash ; }
icosahedron-source(){   echo ${BASH_SOURCE:-$(env-home)/$(icosahedron-src)} ; }
icosahedron-vi(){       vi $(icosahedron-source) ; }
icosahedron-env(){      elocal- ; }
icosahedron-usage(){ cat << EOU


* http://donhavey.com/blog/tutorials/tutorial-3-the-icosahedron-sphere/

* http://sarvanz.blogspot.tw/2013/07/sphere-triangulation-using-icosahedron.html



EOU
}
icosahedron-dir(){ echo $(env-home)/graphics/geometry/icosahedron ; }
icosahedron-tdir(){ echo /tmp/icosahedron ; }
icosahedron-cd(){  cd $(icosahedron-dir); }
icosahedron-get(){
   local dir=$(dirname $(icosahedron-dir)) &&  mkdir -p $dir && cd $dir
}


icosahedron-make()
{
    local tmp=$(icosahedron-tdir)

    mkdir -p $tmp

    icosahedron-cd

    clang icosahedron.cc -c -I. -o $tmp/icosahedron.o

    npy-

    glm-

    clang icosahedron_wrap.cc -c  -I. -I$(npy-idir)/include -I$(glm-idir)/..  -o $tmp/icosahedron_wrap.o

    clang $tmp/icosahedron_wrap.o $tmp/icosahedron.o -L$(npy-idir)/lib -lNPY -o $tmp/icosahedron

    DYLD_LIBRARY_PATH=$(npy-idir)/lib $tmp/icosahedron

    python -c "import numpy as np ; print np.load('/tmp/icosahedron.npy') "  
}

