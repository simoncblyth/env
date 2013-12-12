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


* http://tech.enekochan.com/2012/06/10/install-openscenegraph-2-8-3-with-collada-support-in-ubuntu-12-04/


Requirements
-------------

#.  cmake version 2.4.6 or later


G
---

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

Try disabling the ffmpeg plugin by changing src/osgPlugins/CMakeLists.txt::

    199 #IF(FFMPEG_FOUND AND OSG_CPP_EXCEPTIONS_AVAILABLE)
    200 #    ADD_SUBDIRECTORY(ffmpeg)
    201 #ENDIF()

Then::

    osg-cmake
    osg-make    # looks like does a full rebuild after the cmake change


Collada Plugin
----------------

OSG 3.2.0 dae plugin needs COLLADA DOM v1.4.1, according to 

* /usr/local/env/graphics/openscenegraph/OpenSceneGraph-3.2.0/src/osgPlugins/dae/README.txt



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
