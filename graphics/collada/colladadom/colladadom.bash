#
colladadom-src(){      echo graphics/collada/colladadom/colladadom.bash ; }
colladadom-source(){   echo ${BASH_SOURCE:-$(env-home)/$(colladadom-src)} ; }
colladadom-vi(){       vi $(colladadom-source) ; }
colladadom-env(){      elocal- ; }
colladadom-usage(){ cat << EOU

COLLADA DOM
=============

* http://collada.org/mediawiki/index.php/DOM_guide:_Setting_up
* http://collada.org/mediawiki/index.php/Category:COLLADA_DOM
* http://collada.org/mediawiki/index.php/COLLADA_DOM_user_guide
* http://collada.org/mediawiki/index.php/DOM_guide:_Importing_documents

* https://github.com/rdiankov/collada-dom/blob/master/CMakeLists.txt

Versions
---------

* https://svn.code.sf.net/p/collada-dom/code/tags/

  * starts from 2.0, goes to 2.4.0 

* http://sourceforge.net/projects/collada-dom/files/Collada%20DOM/

  * similarly only 2.0 to 2.4 are available, there are older downloads too


Mac+Linux Requirements
-------------------------

* make version 3.81 or higher is required to build COLLADA DOM. 
  make 3.80 will fail with strange error messages. 
* Also, g++ version 3.4 or higher is required
  version 3.3 is known not to be able to build the source code.
* Cg 2.0 Toolkit
* (Linux) OpenGL and GLUT libraries
* (Mac) Xcode 3.11


Yuck windows/PS3 Libs in SVN 
------------------------------

::

    g4pb:collada blyth$ du -hs colladadom 
    257M    colladadom


G
---

::

    g4pb:collada blyth$ colladadom-cmake
    ...
    -- Compiling Collada DOM Version 2.4.0
    -- Using cmake version 2.8
    -- installing to /usr/local
    -- compiling with double float precision
    -- Boost version: 1.47.0
    -- Found the following Boost libraries:
    --   filesystem
    --   system
    -- found boost version: 104700
    -- Found ZLIB: /usr/lib/libz.dylib (found version "1.2.3")
    -- Found LibXml2: /opt/local/lib/libxml2.dylib 
    -- libxml2 found
    -- checking for module 'minizip'
    --   package 'minizip' not found
    -- compiling minizip from sources and linking statically
    -- checking for module 'libpcrecpp'
    --   found libpcrecpp, version 8.31
    -- Looking for C++ include pcrecpp.h
    -- Looking for C++ include pcrecpp.h - found
    -- Configuring done
    -- Generating done
    -- Build files have been written to: /usr/local/env/graphics/collada/colladadom.build
    g4pb:colladadom.build blyth$ 



minizip 64bit issue
~~~~~~~~~~~~~~~~~~~~~

::

    g4pb:colladadom.build blyth$ colladadom-make
    Scanning dependencies of target minizip
    [  0%] Building C object dom/external-libs/minizip-1.1/CMakeFiles/minizip.dir/ioapi.c.o
    /usr/local/env/graphics/collada/colladadom/dom/external-libs/minizip-1.1/ioapi.c: In function 'fopen64_file_func':
    /usr/local/env/graphics/collada/colladadom/dom/external-libs/minizip-1.1/ioapi.c:115: error: 'Dfseeko64' undeclared (first use in this function)
    /usr/local/env/graphics/collada/colladadom/dom/external-libs/minizip-1.1/ioapi.c:115: error: (Each undeclared identifier is reported only once
    /usr/local/env/graphics/collada/colladadom/dom/external-libs/minizip-1.1/ioapi.c:115: error: for each function it appears in.)
    ...

Follow suggestion in https://github.com/rdiankov/collada-dom/issues/3
to split the definitions in the root CMakeLists.txt into five lines::

    add_definitions("-Dfopen64=fopen -Dfseeko64=fseeko -Dfseek64=fseek -Dftell64=ftell -Dftello64=ftello")

succeeds to resolve this issue.

install
~~~~~~~~

::

    g4pb:colladadom.build blyth$ colladadom-install
    colladadom-install is a function
    colladadom-install () 
    { 
        type $FUNCNAME;
        cd_func $(colladadom-bdir);
        sudo make install
    }
    Password:
    [  1%] Built target minizip
    [ 40%] Built target colladadom141
    [ 93%] Built target colladadom150
    [100%] Built target collada-dom
    Install the project...
    -- Install configuration: ""
    -- Installing: /usr/local/lib/pkgconfig/collada-dom.pc
    -- Installing: /usr/local/lib/pkgconfig/collada-dom-150.pc
    -- Installing: /usr/local/lib/pkgconfig/collada-dom-141.pc
    -- Installing: /usr/local/lib/cmake/collada_dom-2.4/collada_dom-config.cmake
    -- Installing: /usr/local/lib/cmake/collada_dom-2.4/collada_dom-config-version.cmake
    -- Installing: /usr/local/include/collada-dom2.4/1.5
    -- Installing: /usr/local/include/collada-dom2.4/1.5/dom
    -- Installing: /usr/local/include/collada-dom2.4/1.5/dom/domAccessor.h
    -- Installing: /usr/local/include/collada-dom2.4/1.5/dom/domAnimation.h
    ...
    -- Installing: /usr/local/include/collada-dom2.4/1.5/dom/domTypes.h
    -- Installing: /usr/local/include/collada-dom2.4/1.5/dom/domVertices.h
    -- Installing: /usr/local/include/collada-dom2.4/1.5/dom/domVisual_scene.h
    -- Installing: /usr/local/include/collada-dom2.4/1.5/dom/domWires.h
    -- Installing: /usr/local/include/collada-dom2.4/1.4
    -- Installing: /usr/local/include/collada-dom2.4/1.4/dom
    -- Installing: /usr/local/include/collada-dom2.4/1.4/dom/domAccessor.h
    -- Installing: /usr/local/include/collada-dom2.4/1.4/dom/domAnimation.h
    ...
    -- Installing: /usr/local/include/collada-dom2.4/1.4/dom/domTypes.h
    -- Installing: /usr/local/include/collada-dom2.4/1.4/dom/domVertices.h
    -- Installing: /usr/local/include/collada-dom2.4/1.4/dom/domVisual_scene.h
    -- Installing: /usr/local/lib/libcollada-dom2.4-dp.2.4.0.dylib
    -- Installing: /usr/local/lib/libcollada-dom2.4-dp.0.dylib
    -- Installing: /usr/local/lib/libcollada-dom2.4-dp.dylib
    -- Installing: /usr/local/include/collada-dom2.4/dae
    -- Installing: /usr/local/include/collada-dom2.4/dae/daeArray.h
    -- Installing: /usr/local/include/collada-dom2.4/dae/daeArrayTypes.h
    ...
    -- Installing: /usr/local/include/collada-dom2.4/dae.h
    -- Installing: /usr/local/include/collada-dom2.4/dom.h
    g4pb:colladadom.build blyth$ 



libs
~~~~~

::

    g4pb:test blyth$ ll /usr/local/lib/libcollada*          
    lrwxr-xr-x  1 root  wheel        28 12 Dec 20:34 /usr/local/lib/libcollada-dom2.4-dp.dylib -> libcollada-dom2.4-dp.0.dylib
    -rwxr-xr-x  1 root  wheel  12982096 12 Dec 20:34 /usr/local/lib/libcollada-dom2.4-dp.2.4.0.dylib
    lrwxr-xr-x  1 root  wheel        32 12 Dec 20:34 /usr/local/lib/libcollada-dom2.4-dp.0.dylib -> libcollada-dom2.4-dp.2.4.0.dylib



cmake
~~~~~~~

* http://www.cmake.org/Wiki/CMake:How_To_Find_Libraries

::

    -- Installing: /usr/local/lib/cmake/collada_dom-2.4/collada_dom-config.cmake
    -- Installing: /usr/local/lib/cmake/collada_dom-2.4/collada_dom-config-version.cmake
 

pkg-config
~~~~~~~~~~~~

::

    g4pb:dae blyth$ pkg-config --list-all | grep collada
    g4pb:dae blyth$ PKG_CONFIG_PATH=/usr/local/lib/pkgconfig pkg-config --list-all | grep collada
    collada-dom                         collada-dom - COLLADA Document Object Model (DOM), 1.4 support=%OPT_COLLADA14%, 1.5 support=%OPT_COLLADA15%
    collada-dom-141                     collada14dom - COLLADA 1.4 Document Object Model (DOM)
    collada-dom-150                     collada14dom - COLLADA 1.5 Document Object Model (DOM)

    g4pb:test blyth$ PKG_CONFIG_PATH=/usr/local/lib/pkgconfig pkg-config --libs collada-dom-141
    -L/usr/local/lib -lcollada-dom2.4-dp 

    g4pb:test blyth$ PKG_CONFIG_PATH=/usr/local/lib/pkgconfig pkg-config --cflags collada-dom-141
    -DCOLLADA_DOM_SUPPORT141 -DCOLLADA_DOM_SUPPORT150 -DCOLLADA_DOM_DAEFLOAT_IS64 -DCOLLADA_DOM_USING_141 -I/usr/local/include/collada-dom2.4 -I/usr/local/include/collada-dom2.4/1.4 

    g4pb:test blyth$ PKG_CONFIG_PATH=/usr/local/lib/pkgconfig pkg-config --variable=exec_prefix collada-dom-141
    /usr/local/bin

    g4pb:test blyth$ PKG_CONFIG_PATH=/usr/local/lib/pkgconfig pkg-config --variable=prefix collada-dom-141
    /usr/local

    g4pb:test blyth$ PKG_CONFIG_PATH=/usr/local/lib/pkgconfig pkg-config --variable=libdir collada-dom-141
    /usr/local/lib

    g4pb:test blyth$ PKG_CONFIG_PATH=/usr/local/lib/pkgconfig pkg-config --variable=includedir collada-dom-141
    /usr/local/include/collada-dom2.4





EOU
}
colladadom-dir(){ echo $(local-base)/env/graphics/collada/colladadom ; }
colladadom-bdir(){ echo $(colladadom-dir).build ; }
colladadom-sdir(){ echo $(env-home)/graphics/collada/colladadom ; }
colladadom-scd(){  cd $(colladadom-sdir)/$1; }
colladadom-cd(){  cd $(colladadom-dir)/$1; }
colladadom-mate(){ mate $(colladadom-dir) ; }
colladadom-get(){
   local dir=$(dirname $(colladadom-dir)) &&  mkdir -p $dir && cd $dir
   [ ! -d colladadom ] && svn co https://collada-dom.svn.sourceforge.net/svnroot/collada-dom/trunk colladadom
}


colladadom-cmake(){
   local bdir=$(colladadom-bdir)
   [ ! -d $bdir ] && mkdir -p $bdir 
   cd $(colladadom-bdir)
   cmake $(colladadom-dir) 
}

colladadom-make(){
   cd $(colladadom-bdir)
   make $*
}
colladadom-install(){
   type $FUNCNAME
   cd $(colladadom-bdir)
   sudo make install
}

colladadom-test(){
   colladadom-scd testColladaDOM
   ./fromscratch.sh 
   ./build/testColladaDOM
}



