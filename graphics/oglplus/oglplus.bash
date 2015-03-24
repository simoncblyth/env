# === func-gen- : graphics/oglplus/oglplus fgp graphics/oglplus/oglplus.bash fgn oglplus fgh graphics/oglplus
oglplus-src(){      echo graphics/oglplus/oglplus.bash ; }
oglplus-source(){   echo ${BASH_SOURCE:-$(env-home)/$(oglplus-src)} ; }
oglplus-vi(){       vi $(oglplus-source) ; }
oglplus-env(){      elocal- ; }
oglplus-usage(){ cat << EOU

OGLplus
=========

* http://oglplus.org/oglplus/html/index.html



Failing to find GLFW3
-----------------------

* https://github.com/matus-chochlik/oglplus/issues/67

::

    oglplus_common_find_module(GLFW3 glfw3 GLFW/glfw3.h glfw3)

    macro(oglplus_common_find_module PREFIX PC_NAME HEADER LIBRARY)


OGLPlus pkg-config
---------------------

* OGLPlus cmake demands use of the PC approach
* configure.py builds PKG_CONFIG_PATH based on search paths 
* http://www.cmake.org/cmake/help/v3.0/module/FindPkgConfig.html



OGLPlus GLFW
-------------

::

    delta:oglplus-0.59.0 blyth$ find . -type f | grep glfw
    ./example/oglplus/dependencies/glfw3_main.txt
    ./example/oglplus/dependencies/glfw_main.txt
    ./example/oglplus/glfw3_main.cpp
    ./example/oglplus/glfw_main.cpp
    ./example/standalone/029_mandelbrot_glfw3.cpp
    ./test/oglplus/dependencies/fixture_glfw.txt
    ./test/oglplus/fixture_glfw.cpp




EOU
}
oglplus-dir(){  echo $(local-base)/env/graphics/oglplus/$(oglplus-name) ; }
oglplus-idir(){ echo $(oglplus-dir).install ; }
oglplus-bdir(){ echo $(oglplus-dir).build ; }

oglplus-cd(){   cd $(oglplus-dir); }
oglplus-icd(){  cd $(oglplus-idir); }
oglplus-bcd(){  cd $(oglplus-bdir); }

oglplus-get(){
   local dir=$(dirname $(oglplus-dir)) &&  mkdir -p $dir && cd $dir

   local url=$(oglplus-url)
   local tgz=$(basename $url)
   local nam=${tgz/.tar.gz}
   [ ! -f "$tgz" ] && curl -L -O $url
   [ ! -d "$nam" ] && tar zxvf $tgz
}

oglplus-aver(){ echo 0.59.x ; }
oglplus-bver(){ echo 0.59.0 ; }
oglplus-name(){ echo oglplus-$(oglplus-bver) ; }
oglplus-url(){ echo http://downloads.sourceforge.net/project/oglplus/oglplus-$(oglplus-aver)/$(oglplus-name).tar.gz ; }

oglplus-configure-help(){
   local iwd=$PWD
   oglplus-cd 
   python configure.py $*
   cd $iwd
}



oglplus-configure(){
   local iwd=$PWD


   oglplus-cd 

   glfw-
   python configure.py \
          --prefix $(oglplus-idir) --build-dir $(oglplus-bdir) \
          --use-gl-init-lib=GLFW3 \
          --debug-config  \
          --search-dir $(glfw-idir) 
 
   cd $iwd
}



