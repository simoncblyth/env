# === func-gen- : graphics/openscenegraph/osg fgp graphics/openscenegraph/osg.bash fgn osg fgh graphics/openscenegraph
osg-src(){      echo graphics/openscenegraph/osg.bash ; }
osg-source(){   echo ${BASH_SOURCE:-$(env-home)/$(osg-src)} ; }
osg-vi(){       vi $(osg-source) ; }
osg-env(){      elocal- ; }
osg-usage(){ cat << EOU

OPEN SCENE GRAPH
==================

* http://www.openscenegraph.org/index.php/download-section/stable-releases
* http://www.openscenegraph.org/index.php/documentation/10-getting-started
* https://github.com/openscenegraph/osg
* http://trac.macports.org/search?q=OpenSceneGraph

* http://tech.enekochan.com/2012/06/10/install-openscenegraph-2-8-3-with-collada-support-in-ubuntu-12-04/



Requirements
-------------

#.  cmake version 2.4.6 or later


G
---

Macports
~~~~~~~~~

COLLADA seems commented for both OpenSceneGraph and OpenSceneGraph-devel

* http://trac.macports.org/browser/trunk/dports/graphics/OpenSceneGraph
* http://trac.macports.org/browser/trunk/dports/graphics/OpenSceneGraph/Portfile
* http://trac.macports.org/browser/trunk/dports/graphics/OpenSceneGraph/files/patch-CMakeLists.txt.diff
* http://trac.macports.org/browser/trunk/dports/graphics/OpenSceneGraph-devel
* http://trac.macports.org/browser/trunk/dports/graphics/OpenSceneGraph-devel/files/patch-CMakeLists.txt.diff

::

    g4pb:OpenSceneGraph-3.2.0.build blyth$ port info openscenegraph
    OpenSceneGraph @3.2.0 (graphics)
    Variants:             debug, universal

    Description:          OpenSceneGraph is a high-performance 3D graphics toolkit useful in fields such as visual simulation, games, virtual reality, scientific visualization and modelling.
    Homepage:             http://www.openscenegraph.org/

    Extract Dependencies: unzip
    Build Dependencies:   cmake, pkgconfig
    Library Dependencies: freetype, jasper, openexr, zlib, gdal, curl, ffmpeg, poppler, librsvg, giflib, tiff, qt4-mac, boost
    Conflicts with:       OpenSceneGraph-devel
    Platforms:            darwin
    License:              wxWidgets-3
    Maintainers:          nomaintainer@macports.org


Manual cmake Attempt
~~~~~~~~~~~~~~~~~~~~~


::

    g4pb:OpenSceneGraph-3.2.0.build blyth$ osg-
    g4pb:OpenSceneGraph-3.2.0.build blyth$ osg-cmake
    === osg-cmake : running cmake from /usr/local/env/graphics/openscenegraph/OpenSceneGraph-3.2.0.build
    ...
    -- Found Threads: TRUE 
    -- Found OpenGL: /System/Library/Frameworks/OpenGL.framework 
    -- Looking for XOpenDisplay in /opt/local/lib/libX11.dylib;/opt/local/lib/libXext.dylib
    -- Looking for XOpenDisplay in /opt/local/lib/libX11.dylib;/opt/local/lib/libXext.dylib - found
    -- Found X11: /opt/local/lib/libX11.dylib
    -- Found LibXml2: /opt/local/lib/libxml2.dylib 
    -- checking for module 'gta'
    --   package 'gta' not found
    -- Found CURL: /usr/lib/libcurl.dylib 
    -- checking for module 'cairo'
    --   found cairo, version 1.12.2
    -- checking for module 'poppler-glib'
    --   found poppler-glib, version 0.24.1
    -- Performing Test POPPLER_HAS_CAIRO
    -- Performing Test POPPLER_HAS_CAIRO - Success
    -- checking for module 'librsvg-2.0'
    --   package 'librsvg-2.0' not found
    -- checking for module 'gtk+-2.0'
    --   package 'gtk+-2.0' not found
    -- checking for module 'gtkglext-x11-1.0'
    --   package 'gtkglext-x11-1.0' not found
    -- Boost version: 1.47.0
    -- Looking for Q_WS_X11
    -- Looking for Q_WS_X11 - not found.
    -- Looking for Q_WS_WIN
    -- Looking for Q_WS_WIN - not found.
    -- Looking for Q_WS_QWS
    -- Looking for Q_WS_QWS - not found.
    -- Looking for Q_WS_MAC
    -- Looking for Q_WS_MAC - found
    -- Looking for QT_MAC_USE_COCOA
    -- Looking for QT_MAC_USE_COCOA - not found.
    -- Found Qt4: /opt/local/bin/qmake (found version "4.8.5")
    -- Found TIFF: /opt/local/lib/libtiff.dylib 
    AVFoundation disabled for SDK < 10.8
    ...
    The build system is configured to install libraries to /usr/local/lib
    Your applications may not be able to find your installed libraries unless you:
       set your LD_LIBRARY_PATH (user specific) or
       update your ld.so configuration (system wide)



FFmpegDecoderAudio compilation issue
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Fails to compile src/osgPlugins/ffmpeg/FFmpegDecoderAudio.cpp::

    g4pb:OpenSceneGraph-3.2.0.build blyth$ make
    [  0%] Built target OpenThreads
    ...
    [ 97%] Building CXX object src/osgPlugins/txp/CMakeFiles/osgdb_txp.dir/TXPParser.cpp.o
    [ 97%] Building CXX object src/osgPlugins/txp/CMakeFiles/osgdb_txp.dir/TXPSeamLOD.cpp.o
    Linking CXX shared module ../../../lib/osgPlugins-3.2.0/osgdb_txp.so
    [ 97%] Built target osgdb_txp
    Scanning dependencies of target osgdb_ffmpeg
    [ 98%] Building CXX object src/osgPlugins/ffmpeg/CMakeFiles/osgdb_ffmpeg.dir/FFmpegClocks.cpp.o
    /opt/local/include/libavcodec/avcodec.h:2349: warning: 'ImgReSampleContext' is deprecated (declared at /opt/local/include/libavcodec/avcodec.h:2343)
    /opt/local/include/libavcodec/avcodec.h:2359: warning: 'ImgReSampleContext' is deprecated (declared at /opt/local/include/libavcodec/avcodec.h:2343)
    [ 98%] Building CXX object src/osgPlugins/ffmpeg/CMakeFiles/osgdb_ffmpeg.dir/FFmpegDecoderAudio.cpp.o
    /opt/local/include/libavcodec/avcodec.h:2349: warning: 'ImgReSampleContext' is deprecated (declared at /opt/local/include/libavcodec/avcodec.h:2343)
    /opt/local/include/libavcodec/avcodec.h:2359: warning: 'ImgReSampleContext' is deprecated (declared at /opt/local/include/libavcodec/avcodec.h:2343)
    /usr/local/env/graphics/openscenegraph/OpenSceneGraph-3.2.0/src/osgPlugins/ffmpeg/FFmpegDecoderAudio.cpp: In member function 'void osgFFmpeg::FFmpegDecoderAudio::open(AVStream*)':
    /usr/local/env/graphics/openscenegraph/OpenSceneGraph-3.2.0/src/osgPlugins/ffmpeg/FFmpegDecoderAudio.cpp:96: error: 'SAMPLE_FMT_DBL' was not declared in this scope
    /usr/local/env/graphics/openscenegraph/OpenSceneGraph-3.2.0/src/osgPlugins/ffmpeg/FFmpegDecoderAudio.cpp:117: error: 'avcodec_open2' was not declared in this scope
    make[2]: *** [src/osgPlugins/ffmpeg/CMakeFiles/osgdb_ffmpeg.dir/FFmpegDecoderAudio.cpp.o] Error 1
    make[1]: *** [src/osgPlugins/ffmpeg/CMakeFiles/osgdb_ffmpeg.dir/all] Error 2
    make: *** [all] Error 2
    g4pb:OpenSceneGraph-3.2.0.build blyth$ 


* https://github.com/openscenegraph/osg/issues/6
* http://trac.macports.org/ticket/38167

Try disabling the ffmpeg plugin by changing src/osgPlugins/CMakeLists.txt::

    199 #IF(FFMPEG_FOUND AND OSG_CPP_EXCEPTIONS_AVAILABLE)
    200 #    ADD_SUBDIRECTORY(ffmpeg)
    201 #ENDIF()

Then::

    osg-cmake
    osg-make    # Ouch : does a full rebuild after the small cmake change


osgPlugins-3.2.0/osgdb_jp2.so wrong architecture issue
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


::

    [ 85%] Built target osgdb_ktx
    [ 85%] Building CXX object src/osgPlugins/jp2/CMakeFiles/osgdb_jp2.dir/ReaderWriterJP2.cpp.o
    Linking CXX shared module ../../../lib/osgPlugins-3.2.0/osgdb_jp2.so
    ld warning: in /opt/local/lib/libjasper.dylib, file is not of required architecture
    Undefined symbols for architecture i386:
      "_jas_image_destroy", referenced from:
          ReaderWriterJP2::readImage(std::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, osgDB::Options const*) constin ReaderWriterJP2.cpp.o
          ReaderWriterJP2::readImage(std::basic_istream<char, std::char_traits<char> >&, osgDB::Options const*) constin ReaderWriterJP2.cpp.o
          ReaderWriterJP2::writeImage(osg::Image const&, std::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, osgDB::Options const*) constin ReaderWriterJP2.cpp.o
          ReaderWriterJP2::writeImage(osg::Image const&, std::basic_ostream<char, std::char_traits<char> >&, osgDB::Options const*) constin ReaderWriterJP2.cpp.o
          ...
          ReaderWriterJP2::readImage(std::basic_istream<char, std::char_traits<char> >&, osgDB::Options const*) constin ReaderWriterJP2.cpp.o
    ld: symbol(s) not found for architecture i386
    collect2: ld returned 1 exit status
    lipo: can't open input file: /var/folders/Ec/EcFzLXYBGBmHyBnf4ZxShk+++TM/-Tmp-//ccC80UsE.out (No such file or directory)
    make[2]: *** [lib/osgPlugins-3.2.0/osgdb_jp2.so] Error 1
    make[1]: *** [src/osgPlugins/jp2/CMakeFiles/osgdb_jp2.dir/all] Error 2
    make: *** [all] Error 2


* http://trac.macports.org/ticket/18112


Collada Plugin
----------------

OSG 3.2.0 dae plugin needs COLLADA DOM v1.4.1, according to 

* /usr/local/env/graphics/openscenegraph/OpenSceneGraph-3.2.0/src/osgPlugins/dae/README.txt
* 141 is the schema version, not the project version


EOU
}
osg-sdir(){ echo $(local-base)/env/graphics/openscenegraph/$(osg-name) ; }
osg-bdir(){  echo $(osg-sdir).build ; }
osg-cd(){   cd $(osg-bdir); }
osg-scd(){  cd $(osg-sdir); }
osg-name(){ echo OpenSceneGraph-3.2.0 ; }
osg-mate(){ mate $(osg-dir) ; }
osg-get(){
   local dir=$(dirname $(osg-sdir)) &&  mkdir -p $dir && cd $dir

   local nam=$(osg-name)
   local zip=$nam.zip
   local url=http://www.openscenegraph.org/downloads/developer_releases/$zip

   [ ! -f $zip ] && curl -L -O $url
   [ ! -d $nam ] && unzip $zip

   local bld=$(basename $(osg-bdir))
   [ ! -d $bld ] && mkdir -p $bld
}

osg-cmake(){
   local msg="=== $FUNCNAME :"
   cd $(osg-bdir)
   echo $msg running cmake from $PWD
   local nam=$(osg-name)
   cmake ../$nam -DCMAKE_BUILD_TYPE=Release
}

osg-make(){
   cd $(osg-bdir)
   echo $msg running make from $PWD
   make
}
