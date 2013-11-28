# === func-gen- : graphics/meshlabdev/meshlabdev fgp graphics/meshlabdev/meshlabdev.bash fgn meshlabdev fgh graphics/meshlabdev
meshlabdev-src(){      echo graphics/meshlabdev/meshlabdev.bash ; }
meshlabdev-source(){   echo ${BASH_SOURCE:-$(env-home)/$(meshlabdev-src)} ; }
meshlabdev-vi(){       vi $(meshlabdev-source) ; }
meshlabdev-env(){      elocal- ; }
meshlabdev-usage(){ cat << EOU

MESHLABDEV
============

Add hoc approach to meshlab dev over in meshlab- is 
getting unmanageable.  


issues
-------


::

    g++ -c -pipe -O2 -Wall -W -D_REENTRANT -fPIC -DGLEW_STATIC -DQT_NO_DEBUG -DQT_SCRIPT_LIB -DQT_XMLPATTERNS_LIB -DQT_XML_LIB -DQT_OPENGL_LIB -DQT_GUI_LIB -DQT_CORE_LIB -DQT_SHARED -I/usr/local/Trolltech/Qt-4.8.4/mkspecs/linux-g++ -I. -I/usr/local/Trolltech/Qt-4.8.4/include/QtCore -I/usr/local/Trolltech/Qt-4.8.4/include/QtGui -I/usr/local/Trolltech/Qt-4.8.4/include/QtOpenGL -I/usr/local/Trolltech/Qt-4.8.4/include/QtXml -I/usr/local/Trolltech/Qt-4.8.4/include/QtXmlPatterns -I/usr/local/Trolltech/Qt-4.8.4/include/QtScript -I/usr/local/Trolltech/Qt-4.8.4/include -I../.. -I../../../vcglib -I../external/glew-1.7.0/include -I. -I../external/jhead-2.95 -I/usr/X11R6/include -I. -o meshlabdocumentbundler.o meshlabdocumentbundler.cpp
    In file included from ../../../vcglib/wrap/io_trimesh/import_out.h:28,
                     from meshlabdocumentbundler.h:4,
                     from meshlabdocumentbundler.cpp:11:
    ../../../vcglib/vcg/complex/allocate.h:24:2: error: #error "This file should not be included alone. It is automatically included by complex.h"
    In file included from ../../../vcglib/wrap/io_trimesh/import_nvm.h:28,
                     from meshlabdocumentbundler.h:5,
                     from meshlabdocumentbundler.cpp:11:
    ../../../vcglib/vcg/complex/allocate.h:24:2: error: #error "This file should not be included alone. It is automatically included by complex.h"
    make[1]: *** [meshlabdocumentbundler.o] Error 1
    make[1]: Leaving directory `/data1/env/local/env/graphics/meshlabdev/meshlab_trunk/meshlab/src/common'
    make: *** [sub-common-make_default-ordered] Error 2




EOU
}
meshlabdev-dir(){ echo $(local-base)/env/graphics/meshlabdev/meshlab_trunk/meshlab/src ; }
meshlabdev-fold(){  echo $(dirname $(dirname $(dirname $(meshlabdev-dir))))  ; }
meshlabdev-cd(){  cd $(meshlabdev-dir); }
meshlabdev-mate(){ mate $(meshlabdev-dir) ; }
meshlabdev-get(){
   local dir=$(meshlabdev-fold) &&  mkdir -p $dir && cd $dir
   svn checkout http://svn.code.sf.net/p/meshlab/code/trunk/  meshlab_trunk
   svn checkout http://svn.code.sf.net/p/vcg/code/trunk/ vcglib_trunk

   echo mimic the tarballs layout 
   ( cd meshlab_trunk ; ln -s ../vcglib_trunk/vcglib vcglib ) 
}

meshlabdev-git(){
   echo TEST BUILDING SVN CHECKOUT BEFORE GIT SVN CLONE : AS LIABLE TO TAKE DAYS TO DOWNLOAD ENTIRE REPO HISTORY
   git svn clone http://svn.code.sf.net/p/meshlab/code/trunk/  meshlab_trunk
   git svn clone http://svn.code.sf.net/p/vcg/code/trunk/ vcglib_trunk
}
