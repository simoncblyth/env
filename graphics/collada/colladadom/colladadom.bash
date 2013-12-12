#
colladadom-src(){      echo graphics/collada/colladadom/colladadom.bash ; }
colladadom-source(){   echo ${BASH_SOURCE:-$(env-home)/$(colladadom-src)} ; }
colladadom-vi(){       vi $(colladadom-source) ; }
colladadom-env(){      elocal- ; }
colladadom-usage(){ cat << EOU

COLLADA DOM
=============

* http://collada.org/mediawiki/index.php/DOM_guide:_Setting_up

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




EOU
}
colladadom-dir(){ echo $(local-base)/env/graphics/collada/colladadom ; }
colladadom-bdir(){ echo $(colladadom-dir).build ; }
colladadom-cd(){  cd $(colladadom-dir); }
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
   make
}
