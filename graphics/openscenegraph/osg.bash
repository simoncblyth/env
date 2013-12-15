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


Versions
---------

* https://github.com/openscenegraph/osg
* https://github.com/openscenegraph/osg/tree/OpenSceneGraph-3.2
* https://github.com/openscenegraph/osg/commits/OpenSceneGraph-3.2


Debugging
---------

::

     OSG_NOTIFY_LEVEL=DEBUG osgviewer 2.dae 

::

    FindFileInPath() : USING /usr/local/lib/osgPlugins-3.2.0/osgdb_dae.so
    Opened DynamicLibrary osgPlugins-3.2.0/osgdb_dae.so
    ReaderWriterDAE( "2.dae" )
    Load failed in COLLADA DOM
    Load failed in COLLADA DOM conversion
    Warning: Error in reading to "2.dae".
    osgviewer: No data loaded
    Viewer::~Viewer():: start destructor getThreads = 0

::

    g4pb:src blyth$ find . -name '*.cpp' -exec grep -H "Load failed in" {} \;
    ./osgPlugins/dae/daeReader.cpp:        OSG_WARN << "Load failed in COLLADA DOM" << std::endl;
    ./osgPlugins/dae/ReaderWriterDAE.cpp:        OSG_WARN << "Load failed in COLLADA DOM conversion" << std::endl;
    ./osgPlugins/dae/ReaderWriterDAE.cpp:        OSG_WARN << "Load failed in COLLADA DOM conversion" << std::endl;


osgPlugins/dae/ReaderWriterDAE.cpp::

    138     OSG_INFO << "ReaderWriterDAE( \"" << fileName << "\" )" << std::endl;
    139 
    140     if (NULL == pDAE)
    141     {
    142         bOwnDAE = true;
    143         //pDAE = new DAE;
    144         pDAE = new DAE(NULL,NULL,"1.4.1");  // SCB be specific as required from 2.4.0
    145     }
    146     std::auto_ptr<DAE> scopedDae(bOwnDAE ? pDAE : NULL);        // Deallocates locally created structure at scope exit
    147 
    148     osgDAE::daeReader daeReader(pDAE, &pluginOptions);
    149 
    150     // Convert file name to URI
    151     std::string fileURI = ConvertFilePathToColladaCompatibleURI(fileName);
    152 
    153     if ( ! daeReader.convert( fileURI ) )
    154     {
    155         OSG_WARN << "Load failed in COLLADA DOM conversion" << std::endl;
    156         return ReadResult::ERROR_IN_READING_FILE;
    157     }




OSG Namespaces
-----------------

* http://forum.openscenegraph.org/viewtopic.php?t=7733


osgviewer
----------

* http://trac.openscenegraph.org/projects/osg//wiki/Support/UserGuides/osgviewer

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


Change /usr/local/env/graphics/openscenegraph/OpenSceneGraph-3.2.0/src/osgPlugins/CMakeLists.txt 
commenting out jp2, then "make" from the build dir succeeds to not re-do everything, rapidly after 
approx 1 min getting to the next issue.


tiff architecture
~~~~~~~~~~~~~~~~~~~~

::

    [ 84%] Building CXX object src/osgPlugins/tiff/CMakeFiles/osgdb_tiff.dir/ReaderWriterTIFF.cpp.o
    Linking CXX shared module ../../../lib/osgPlugins-3.2.0/osgdb_tiff.so
    ld warning: in /opt/local/lib/libtiff.dylib, file is not of required architecture
    Undefined symbols for architecture i386:
      "_TIFFReadScanline", referenced from:
          simage_tiff_load(std::basic_istream<char, std::char_traits<char> >&, int&, int&, int&, unsigned short&)in ReaderWriterTIFF.cpp.o


quicktime plugin many deprecated warnings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    Scanning dependencies of target osgdb_qt
    [ 98%] Building CXX object src/osgPlugins/quicktime/CMakeFiles/osgdb_qt.dir/MovieData.cpp.o
    /usr/local/env/graphics/openscenegraph/OpenSceneGraph-3.2.0/src/osgPlugins/quicktime/MovieData.cpp: In destructor 'MovieData::~MovieData()':
    /usr/local/env/graphics/openscenegraph/OpenSceneGraph-3.2.0/src/osgPlugins/quicktime/MovieData.cpp:29: warning: 'DisposeGWorld' is deprecated (declared at /Developer/SDKs/MacOSX10.5.sdk/System/Library/Frameworks/ApplicationServices.framework/Frameworks/QD.framework/Headers/QDOffscreen.h:230)
    ...
    /usr/local/env/graphics/openscenegraph/OpenSceneGraph-3.2.0/src/osgPlugins/quicktime/MovieData.cpp:197: warning: 'GetGWorld' is deprecated (declared at /Developer/SDKs/MacOSX10.5.sdk/System/Library/Frameworks/ApplicationServices.framework/Frameworks/QD.framework/Headers/QDOffscreen.h:244)
    ...
    /usr/local/env/graphics/openscenegraph/OpenSceneGraph-3.2.0/src/osgPlugins/quicktime/QuicktimeLiveImageStream.cpp:559: warning: 'GetGWorld' is deprecated (declared at /Developer/SDKs/MacOSX10.5.sdk/System/Library/Frameworks/ApplicationServices.framework/Frameworks/QD.framework/Headers/QDOffscreen.h:244)
    [ 98%] Building CXX object src/osgPlugins/quicktime/CMakeFiles/osgdb_qt.dir/ReaderWriterQT.cpp.o
    Linking CXX shared module ../../../lib/osgPlugins-3.2.0/osgdb_qt.so
    [ 98%] Built target osgdb_qt

qtkit fails
~~~~~~~~~~~~~

::

    Scanning dependencies of target osgdb_QTKit
    [ 98%] Building CXX object src/osgPlugins/QTKit/CMakeFiles/osgdb_QTKit.dir/ReaderWriterQTKit.cpp.o
    [ 98%] Building CXX object src/osgPlugins/QTKit/CMakeFiles/osgdb_QTKit.dir/OSXQTKitVideo.mm.o
    /usr/local/env/graphics/openscenegraph/OpenSceneGraph-3.2.0/src/osgPlugins/QTKit/OSXQTKitVideo.mm: In static member function 'static void OSXQTKitVideo::initializeQTKit()':
    /usr/local/env/graphics/openscenegraph/OpenSceneGraph-3.2.0/src/osgPlugins/QTKit/OSXQTKitVideo.mm:100: error: 'dispatch_get_main_queue' was not declared in this scope
    /usr/local/env/graphics/openscenegraph/OpenSceneGraph-3.2.0/src/osgPlugins/QTKit/OSXQTKitVideo.mm:100: error: expected primary-expression before '^' token
    ...
    /usr/local/env/graphics/openscenegraph/OpenSceneGraph-3.2.0/src/osgPlugins/QTKit/OSXQTKitVideo.mm:390: error: 'kCVPixelBufferLock_ReadOnly' was not declared in this scope
    lipo: can't figure out the architecture type of: /var/folders/Ec/EcFzLXYBGBmHyBnf4ZxShk+++TM/-Tmp-//ccajBypt.out
    make[2]: *** [src/osgPlugins/QTKit/CMakeFiles/osgdb_QTKit.dir/OSXQTKitVideo.mm.o] Error 1
    make[1]: *** [src/osgPlugins/QTKit/CMakeFiles/osgdb_QTKit.dir/all] Error 2
    make: *** [all] Error 2

freetype fails
~~~~~~~~~~~~~~

::

    Scanning dependencies of target osgdb_freetype
    [ 97%] Building CXX object src/osgPlugins/freetype/CMakeFiles/osgdb_freetype.dir/FreeTypeFont.cpp.o
    [ 97%] Building CXX object src/osgPlugins/freetype/CMakeFiles/osgdb_freetype.dir/FreeTypeLibrary.cpp.o
    [ 97%] Building CXX object src/osgPlugins/freetype/CMakeFiles/osgdb_freetype.dir/ReaderWriterFreeType.cpp.o
    Linking CXX shared module ../../../lib/osgPlugins-3.2.0/osgdb_freetype.so
    ld warning: in /opt/local/lib/libfreetype.dylib, file is not of required architecture
    Undefined symbols for architecture i386:
      "_FT_New_Face", referenced from:
          FreeTypeLibrary::getFace(std::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, unsigned int, FT_FaceRec_*&)in FreeTypeLibrary.cpp.o
      "_FT_Get_Kerning", referenced from:
          FreeTypeFont::getKerning(unsigned int, unsigned int, osgText::KerningType)in FreeTypeFont.cpp.o


Qt fails
~~~~~~~~~~

::

    [ 96%] Building CXX object src/osgQt/CMakeFiles/osgQt.dir/__/__/include/osgQt/moc_QGraphicsViewAdapter.cxx.o
    Linking CXX shared library ../../lib/libosgQt.dylib
    ld warning: in /opt/local/lib/libQtCore.dylib, file is not of required architecture
    ld warning: in /opt/local/lib/libQtGui.dylib, file is not of required architecture
    ld warning: in /opt/local/lib/libQtOpenGL.dylib, file is not of required architecture
    Undefined symbols:
      "QCursor::QCursor(Qt::CursorShape)", referenced from:
          osgQt::GraphicsWindowQt::useCursor(bool) in GraphicsWindowQt.cpp.o
          osgQt::GraphicsWindowQt::setCursor(osgViewer::GraphicsWindow::MouseCursor)       in GraphicsWindowQt.cpp.o
          osgQt::GraphicsWindowQt::setCursor(osgViewer::GraphicsWindow::MouseCursor)       in GraphicsWindowQt.cpp.o


PDF fails
~~~~~~~~~

::

    Scanning dependencies of target osgdb_pdf
    [ 97%] Building CXX object src/osgPlugins/pdf/CMakeFiles/osgdb_pdf.dir/ReaderWriterPDF.cpp.o
    Linking CXX shared module ../../../lib/osgPlugins-3.2.0/osgdb_pdf.so
    ld warning: in /opt/local/lib/libcairo.dylib, file is not of required architecture
    ld warning: in /opt/local/lib/libpoppler-glib.dylib, file is not of required architecture
    ld warning: in /opt/local/lib/libgio-2.0.dylib, file is not of required architecture
    ld warning: in /opt/local/lib/libgobject-2.0.dylib, file is not of required architecture
    ld warning: in /opt/local/lib/libglib-2.0.dylib, file is not of required architecture
    ld warning: in /opt/local/lib/libintl.dylib, file is not of required architecture
    Undefined symbols for architecture i386:
      "_g_object_unref", referenced from:
          PopplerPdfImage::~PopplerPdfImage()in ReaderWriterPDF.cpp.o
          PopplerPdfImage::~PopplerPdfImage()in ReaderWriterPDF.cpp.o
          PopplerPdfImage::open(std::basic_string<char, std::char_traits<char>, std::allocator<char> > const&)in ReaderWriterPDF.cpp.o
      "_g_type_init", referenced from:
          PopplerPdfImage::open(std::basic_string<char, std::char_traits<char>, std::allocator<char> > const&)in ReaderWriterPDF.cpp.o
      "_poppler_page_get_size", referenced from:


SDL fails
~~~~~~~~~~~

::

    [ 99%] Building CXX object src/osgPlugins/sdl/CMakeFiles/osgdb_sdl.dir/ReaderWriterSDL.cpp.o
    Linking CXX shared module ../../../lib/osgPlugins-3.2.0/osgdb_sdl.so
    /usr/bin/c++   -mmacosx-version-min=10.5 -ftree-vectorize -fvisibility-inlines-hidden -O3 -DNDEBUG -arch ppc -arch i386 -isysroot /Developer/SDKs/MacOSX10.5.sdk -bundle -Wl,-headerpad_max_install_names   -o ../../../lib/osgPlugins-3.2.0/osgdb_sdl.so CMakeFiles/osgdb_sdl.dir/JoystickDevice.cpp.o CMakeFiles/osgdb_sdl.dir/ReaderWriterSDL.cpp.o ../../../lib/libOpenThreads.3.2.0.dylib ../../../lib/libosg.3.2.0.dylib ../../../lib/libosgDB.3.2.0.dylib ../../../lib/libosgUtil.3.2.0.dylib ../../../lib/libosgGA.3.2.0.dylib -framework SDL -framework Cocoa ../../../lib/libosgDB.3.2.0.dylib -framework Carbon /usr/lib/libz.dylib ../../../lib/libosgUtil.3.2.0.dylib ../../../lib/libosg.3.2.0.dylib ../../../lib/libOpenThreads.3.2.0.dylib -lpthread /usr/lib/libm.dylib /usr/lib/libdl.dylib -framework OpenGL 
    ld warning: in /Developer/SDKs/MacOSX10.5.sdk/Library/Frameworks//SDL.framework/SDL, file is not of required architecture
    Undefined symbols for architecture i386:
      "_SDL_JoystickNumHats", referenced from:
          JoystickDevice::JoystickDevice()in JoystickDevice.cpp.o
      "_SDL_JoystickNumAxes", referenced from:

ZEROCONF
~~~~~~~~~~~

::

    Scanning dependencies of target osgdb_zeroconf
    [ 98%] Building CXX object src/osgPlugins/ZeroConfDevice/CMakeFiles/osgdb_zeroconf.dir/AutoDiscovery.cpp.o
    [ 98%] Building CXX object src/osgPlugins/ZeroConfDevice/CMakeFiles/osgdb_zeroconf.dir/ReaderWriterZeroConfDevice.cpp.o
    [ 98%] Building CXX object src/osgPlugins/ZeroConfDevice/CMakeFiles/osgdb_zeroconf.dir/AutoDiscoveryBonjourImpl.mm.o
    /usr/local/env/graphics/openscenegraph/OpenSceneGraph-3.2.0/src/osgPlugins/ZeroConfDevice/AutoDiscoveryBonjourImpl.mm:21: error: cannot find protocol declaration for 'NSNetServiceDelegate'
    /usr/local/env/graphics/openscenegraph/OpenSceneGraph-3.2.0/src/osgPlugins/ZeroConfDevice/AutoDiscoveryBonjourImpl.mm:60: error: cannot find protocol declaration for 'NSNetServiceBrowserDelegate'
    /usr/local/env/graphics/openscenegraph/OpenSceneGraph-3.2.0/src/osgPlugins/ZeroConfDevice/AutoDiscoveryBonjourImpl.mm:60: error: cannot find protocol declaration for 'NSNetServiceDelegate'


osgqt
~~~~~~~

::

    [ 98%] Built target osgdb_trk
    Linking CXX shared library ../../lib/libosgQt.dylib
    ld warning: in /opt/local/lib/libQtCore.dylib, file is not of required architecture
    ld warning: in /opt/local/lib/libQtGui.dylib, file is not of required architecture
    ld warning: in /opt/local/lib/libQtOpenGL.dylib, file is not of required architecture
    Undefined symbols:
      "QCursor::QCursor(Qt::CursorShape)", referenced from:
          osgQt::GraphicsWindowQt::useCursor(bool) in GraphicsWindowQt.cpp.o
          osgQt::GraphicsWindowQt::setCursor(osgViewer::GraphicsWindow::MouseCursor)       in GraphicsWindowQt.cpp.o


cmake architectures
~~~~~~~~~~~~~~~~~~~~~~

From the README.txt::

    097 CMAKE_OSX_ARCHITECTURES - Xcode can create applications, executables,
    098 libraries, and frameworks that can be run on more than one architecture.
    099 Use this setting to indicate the architectures on which to build OSG.
    100 Possibilities include ppc, ppc64, i386, and x86_64. Building OSG using
    101 either of the 64-bit options (ppc64 and x86_64) has its own caveats
    102 below.

Maybe CMAKE_OSX_ARCHITECTURES defaults to something inappropriate for PPC ?

* http://www.cmake.org/pipermail/cmake/2009-September/031845.html
* https://issues.apache.org/jira/secure/attachment/12528323/0001-No-default-for-CMAKE_OSX_ARCHITECTURES.patch

Dumping it::

    -- CMAKE_OSX_ARCHITECTURES : ppci386
    --  MYARCH : ppc
    --  MYARCH : i386

Maybe mixing two-architecture things with one architecture ones from macports is the cause of the problem::

    g4pb:OpenSceneGraph-3.2.0 blyth$ file /usr/local/lib/libOpenThreads.3.2.0.dylib
    /usr/local/lib/libOpenThreads.3.2.0.dylib: Mach-O universal binary with 2 architectures
    /usr/local/lib/libOpenThreads.3.2.0.dylib (for architecture ppc7400):   Mach-O dynamically linked shared library ppc
    /usr/local/lib/libOpenThreads.3.2.0.dylib (for architecture i386):      Mach-O dynamically linked shared library i386

::

    g4pb:OpenSceneGraph-3.2.0 blyth$ file /opt/local/lib/libQtCore.dylib
    /opt/local/lib/libQtCore.dylib: symbolic link to /opt/local/Library/Frameworks/QtCore.framework/QtCore

    g4pb:OpenSceneGraph-3.2.0 blyth$ file /opt/local/Library/Frameworks/QtCore.framework/QtCore
    /opt/local/Library/Frameworks/QtCore.framework/QtCore: symbolic link to Versions/4/QtCore

    g4pb:OpenSceneGraph-3.2.0 blyth$ file /opt/local/Library/Frameworks/QtCore.framework/Versions/4/QtCore
    /opt/local/Library/Frameworks/QtCore.framework/Versions/4/QtCore: Mach-O dynamically linked shared library ppc

    g4pb:OpenSceneGraph-3.2.0 blyth$ file -L /opt/local/lib/libQtCore.dylib
    /opt/local/lib/libQtCore.dylib: Mach-O dynamically linked shared library ppc

cmake argument example::

     -D CMAKE_OSX_ARCHITECTURES:STRING="armv6;armv7" \




* http://trac.macports.org/ticket/18112


Install
---------

::

    Install the project...
    -- Install configuration: "Release"
    -- Installing: /usr/local/lib/pkgconfig/openscenegraph.pc
    -- Installing: /usr/local/lib/pkgconfig/openscenegraph-osg.pc
    -- Installing: /usr/local/lib/pkgconfig/openscenegraph-osgDB.pc
    ...
    -- Installing: /usr/local/lib/pkgconfig/openscenegraph-osgQt.pc
    -- Installing: /usr/local/lib/pkgconfig/openthreads.pc
    -- Installing: /usr/local/lib/libOpenThreads.3.2.0.dylib
    -- Installing: /usr/local/lib/libOpenThreads.13.dylib
    -- Installing: /usr/local/lib/libOpenThreads.dylib
    -- Installing: /usr/local/include/OpenThreads/Atomic
    ...
    -- Installing: /usr/local/lib/libosg.3.2.0.dylib
    -- Installing: /usr/local/lib/libosg.100.dylib
    -- Installing: /usr/local/lib/libosg.dylib
    -- Installing: /usr/local/include/osg/AlphaFunc
    -- Installing: /usr/local/include/osg/AnimationPath
    ...
    -- Installing: /usr/local/include/osg/View
    -- Installing: /usr/local/include/osg/Viewport
    -- Installing: /usr/local/include/osg/Config
    -- Installing: /usr/local/lib/libosgDB.3.2.0.dylib
    -- Installing: /usr/local/lib/libosgDB.100.dylib
    -- Installing: /usr/local/lib/libosgDB.dylib
    -- Installing: /usr/local/include/osgDB/DataTypes
    -- Installing: /usr/local/include/osgDB/StreamOperator
    -- Installing: /usr/local/include/osgDB/Serializer
    ...
    -- Installing: /usr/local/include/osgDB/WriteFile
    -- Installing: /usr/local/include/osgDB/XmlParser
    -- Installing: /usr/local/lib/libosgUtil.3.2.0.dylib
    -- Installing: /usr/local/lib/libosgUtil.100.dylib
    -- Installing: /usr/local/lib/libosgUtil.dylib
    -- Installing: /usr/local/include/osgUtil/ConvertVec
    -- Installing: /usr/local/include/osgUtil/CubeMapGenerator
    -- Installing: /usr/local/include/osgUtil/CullVisitor
    -- Installing: /usr/local/include/osgUtil/DelaunayTriangulator
    ...
    -- Installing: /usr/local/include/osgUtil/TransformCallback
    -- Installing: /usr/local/include/osgUtil/TriStripVisitor
    -- Installing: /usr/local/include/osgUtil/UpdateVisitor
    -- Installing: /usr/local/include/osgUtil/Version
    -- Installing: /usr/local/lib/libosgGA.3.2.0.dylib
    -- Installing: /usr/local/lib/libosgGA.100.dylib
    -- Installing: /usr/local/lib/libosgGA.dylib
    -- Installing: /usr/local/include/osgGA/AnimationPathManipulator
    -- Installing: /usr/local/include/osgGA/DriveManipulator
    ...
    -- Installing: /usr/local/include/osgGA/TrackballManipulator
    -- Installing: /usr/local/include/osgGA/UFOManipulator
    -- Installing: /usr/local/include/osgGA/Version
    -- Installing: /usr/local/include/osgGA/CameraViewSwitchManipulator
    -- Installing: /usr/local/lib/libosgText.3.2.0.dylib
    -- Installing: /usr/local/lib/libosgText.100.dylib
    -- Installing: /usr/local/lib/libosgText.dylib
    -- Installing: /usr/local/include/osgText/Export
    -- Installing: /usr/local/include/osgText/Font
    -- Installing: /usr/local/include/osgText/Font3D
    ...
    -- Installing: /usr/local/include/osgText/Text3D
    -- Installing: /usr/local/include/osgText/Version
    -- Installing: /usr/local/lib/libosgViewer.3.2.0.dylib
    -- Installing: /usr/local/lib/libosgViewer.100.dylib
    -- Installing: /usr/local/lib/libosgViewer.dylib
    -- Installing: /usr/local/include/osgViewer/CompositeViewer
    -- Installing: /usr/local/include/osgViewer/Export
    ...
    -- Installing: /usr/local/include/osgViewer/api/Cocoa/GraphicsHandleCocoa
    -- Installing: /usr/local/include/osgViewer/api/Cocoa/GraphicsWindowCocoa
    -- Installing: /usr/local/include/osgViewer/api/Cocoa/PixelBufferCocoa
    -- Installing: /usr/local/lib/libosgAnimation.3.2.0.dylib
    -- Installing: /usr/local/lib/libosgAnimation.100.dylib
    -- Installing: /usr/local/lib/libosgAnimation.dylib
    -- Installing: /usr/local/include/osgAnimation/Action
    -- Installing: /usr/local/include/osgAnimation/ActionAnimation
    ...
    -- Installing: /usr/local/include/osgAnimation/StackedMatrixElement
    -- Installing: /usr/local/include/osgAnimation/StackedQuaternionElement
    -- Installing: /usr/local/include/osgAnimation/StackedRotateAxisElement
    -- Installing: /usr/local/include/osgAnimation/StackedScaleElement
    -- Installing: /usr/local/include/osgAnimation/StackedTransformElement
    -- Installing: /usr/local/include/osgAnimation/StackedTranslateElement
    -- Installing: /usr/local/include/osgAnimation/StackedTransform
    -- Installing: /usr/local/include/osgAnimation/StatsVisitor
    -- Installing: /usr/local/include/osgAnimation/StatsHandler
    -- Installing: /usr/local/include/osgAnimation/Target
    -- Installing: /usr/local/include/osgAnimation/Timeline
    -- Installing: /usr/local/include/osgAnimation/TimelineAnimationManager
    -- Installing: /usr/local/include/osgAnimation/UpdateBone
    -- Installing: /usr/local/include/osgAnimation/UpdateMaterial
    -- Installing: /usr/local/include/osgAnimation/UpdateMatrixTransform
    -- Installing: /usr/local/include/osgAnimation/Vec3Packed
    -- Installing: /usr/local/include/osgAnimation/VertexInfluence
    -- Installing: /usr/local/lib/libosgFX.3.2.0.dylib
    -- Installing: /usr/local/lib/libosgFX.100.dylib
    -- Installing: /usr/local/lib/libosgFX.dylib
    -- Installing: /usr/local/include/osgFX/AnisotropicLighting
    -- Installing: /usr/local/include/osgFX/BumpMapping
    -- Installing: /usr/local/include/osgFX/Cartoon
    ...
    -- Installing: /usr/local/include/osgFX/Validator
    -- Installing: /usr/local/include/osgFX/Version
    -- Installing: /usr/local/lib/libosgManipulator.3.2.0.dylib
    -- Installing: /usr/local/lib/libosgManipulator.100.dylib
    -- Installing: /usr/local/lib/libosgManipulator.dylib
    -- Installing: /usr/local/include/osgManipulator/AntiSquish
    -- Installing: /usr/local/include/osgManipulator/Command
    -- Installing: /usr/local/include/osgManipulator/CommandManager
    ...
    -- Installing: /usr/local/include/osgManipulator/Translate2DDragger
    -- Installing: /usr/local/include/osgManipulator/TranslateAxisDragger
    -- Installing: /usr/local/include/osgManipulator/TranslatePlaneDragger
    -- Installing: /usr/local/include/osgManipulator/Version
    -- Installing: /usr/local/lib/libosgParticle.3.2.0.dylib
    -- Installing: /usr/local/lib/libosgParticle.100.dylib
    -- Installing: /usr/local/lib/libosgParticle.dylib
    -- Installing: /usr/local/include/osgParticle/AccelOperator
    -- Installing: /usr/local/include/osgParticle/AngularAccelOperator
    -- Installing: /usr/local/include/osgParticle/BoxPlacer
    ...
    -- Installing: /usr/local/include/osgParticle/OrbitOperator
    -- Installing: /usr/local/include/osgParticle/DomainOperator
    -- Installing: /usr/local/include/osgParticle/BounceOperator
    -- Installing: /usr/local/include/osgParticle/SinkOperator
    -- Installing: /usr/local/lib/libosgPresentation.3.2.0.dylib
    -- Installing: /usr/local/lib/libosgPresentation.100.dylib
    -- Installing: /usr/local/lib/libosgPresentation.dylib
    -- Installing: /usr/local/include/osgPresentation/Export
    -- Installing: /usr/local/include/osgPresentation/AnimationMaterial
    -- Installing: /usr/local/include/osgPresentation/CompileSlideCallback
    -- Installing: /usr/local/include/osgPresentation/PickEventHandler
    -- Installing: /usr/local/include/osgPresentation/PropertyManager
    -- Installing: /usr/local/include/osgPresentation/KeyEventHandler
    -- Installing: /usr/local/include/osgPresentation/SlideEventHandler
    -- Installing: /usr/local/include/osgPresentation/SlideShowConstructor
    -- Installing: /usr/local/include/osgPresentation/Timeout
    -- Installing: /usr/local/lib/libosgShadow.3.2.0.dylib
    -- Installing: /usr/local/lib/libosgShadow.100.dylib
    -- Installing: /usr/local/lib/libosgShadow.dylib
    -- Installing: /usr/local/include/osgShadow/Export
    -- Installing: /usr/local/include/osgShadow/OccluderGeometry
    -- Installing: /usr/local/include/osgShadow/ShadowMap
    -- Installing: /usr/local/include/osgShadow/ShadowTechnique
    ...
    -- Installing: /usr/local/include/osgShadow/StandardShadowMap
    -- Installing: /usr/local/include/osgShadow/ViewDependentShadowTechnique
    -- Installing: /usr/local/include/osgShadow/ViewDependentShadowMap
    -- Installing: /usr/local/lib/libosgSim.3.2.0.dylib
    -- Installing: /usr/local/lib/libosgSim.100.dylib
    -- Installing: /usr/local/lib/libosgSim.dylib
    -- Installing: /usr/local/include/osgSim/BlinkSequence
    -- Installing: /usr/local/include/osgSim/ColorRange
    ...
    -- Installing: /usr/local/include/osgSim/Sector
    -- Installing: /usr/local/include/osgSim/ShapeAttribute
    -- Installing: /usr/local/include/osgSim/SphereSegment
    -- Installing: /usr/local/include/osgSim/Version
    -- Installing: /usr/local/include/osgSim/VisibilityGroup
    -- Installing: /usr/local/lib/libosgTerrain.3.2.0.dylib
    -- Installing: /usr/local/lib/libosgTerrain.100.dylib
    -- Installing: /usr/local/lib/libosgTerrain.dylib
    -- Installing: /usr/local/include/osgTerrain/Export
    -- Installing: /usr/local/include/osgTerrain/Locator
    -- Installing: /usr/local/include/osgTerrain/Layer
    -- Installing: /usr/local/include/osgTerrain/TerrainTile
    -- Installing: /usr/local/include/osgTerrain/TerrainTechnique
    -- Installing: /usr/local/include/osgTerrain/Terrain
    -- Installing: /usr/local/include/osgTerrain/GeometryTechnique
    -- Installing: /usr/local/include/osgTerrain/ValidDataOperator
    -- Installing: /usr/local/include/osgTerrain/Version
    -- Installing: /usr/local/lib/libosgWidget.3.2.0.dylib
    -- Installing: /usr/local/lib/libosgWidget.100.dylib
    -- Installing: /usr/local/lib/libosgWidget.dylib
    -- Installing: /usr/local/include/osgWidget/Export
    -- Installing: /usr/local/include/osgWidget/Box
    ...
    -- Installing: /usr/local/include/osgWidget/Window
    -- Installing: /usr/local/include/osgWidget/WindowManager
    -- Installing: /usr/local/lib/libosgVolume.3.2.0.dylib
    -- Installing: /usr/local/lib/libosgVolume.100.dylib
    -- Installing: /usr/local/lib/libosgVolume.dylib
    -- Installing: /usr/local/include/osgVolume/Export
    -- Installing: /usr/local/include/osgVolume/FixedFunctionTechnique
    -- Installing: /usr/local/include/osgVolume/Layer
    -- Installing: /usr/local/include/osgVolume/Locator
    -- Installing: /usr/local/include/osgVolume/Property
    -- Installing: /usr/local/include/osgVolume/RayTracedTechnique
    -- Installing: /usr/local/include/osgVolume/Version
    -- Installing: /usr/local/include/osgVolume/Volume
    -- Installing: /usr/local/include/osgVolume/VolumeTechnique
    -- Installing: /usr/local/include/osgVolume/VolumeTile
    -- Installing: /usr/local/lib/osgPlugins-3.2.0/osgdb_serializers_osg.so
    -- Installing: /usr/local/lib/osgPlugins-3.2.0/osgdb_serializers_osganimation.so
    ...
    -- Installing: /usr/local/lib/osgPlugins-3.2.0/osgdb_osc.so
    -- Installing: /usr/local/lib/osgPlugins-3.2.0/osgdb_trk.so
    -- Installing: /usr/local/bin/osgviewer
    -- Installing: /usr/local/bin/osgarchive
    -- Installing: /usr/local/bin/osgconv
    -- Installing: /usr/local/bin/osgfilecache
    -- Installing: /usr/local/bin/osgversion
    -- Installing: /usr/local/bin/present3D



Attempt to add COLLADA plugin avoiding full rebuild
-----------------------------------------------------


Add the below at the end of /usr/local/env/graphics/openscenegraph/OpenSceneGraph-3.2.0/src/osgPlugins/CMakeLists.txt::

    SET( COLLADA_INCLUDE_DIR /usr/local/include/collada-dom2.4)
    SET( COLLADA_DYNAMIC_LIBRARY /usr/local/lib/libcollada-dom2.4-dp.2.4.0.dylib)
    ADD_SUBDIRECTORY(dae)

And make from the corresponding build dir /usr/local/env/graphics/openscenegraph/OpenSceneGraph-3.2.0.build/src/osgPlugins


dae plugin
~~~~~~~~~~~

::

    Scanning dependencies of target osgdb_dae
    [ 98%] Building CXX object src/osgPlugins/dae/CMakeFiles/osgdb_dae.dir/daeReader.cpp.o
    /usr/local/env/graphics/openscenegraph/OpenSceneGraph-3.2.0/src/osgPlugins/dae/daeReader.h:162: error: 'domUpAxisType' does not name a type
    /usr/local/env/graphics/openscenegraph/OpenSceneGraph-3.2.0/src/osgPlugins/dae/daeReader.h:228: error: 'domFx_opaque_enum' does not name a type
    /usr/local/env/graphics/openscenegraph/OpenSceneGraph-3.2.0/src/osgPlugins/dae/daeReader.h:241: error: 'domMaterial' was not declared in this scope
    /usr/local/env/graphics/openscenegraph/OpenSceneGraph-3.2.0/src/osgPlugins/dae/daeReader.h:241: error: template argument 1 is invalid
    /usr/local/env/graphics/openscenegraph/OpenSceneGraph-3.2.0/src/osgPlugins/dae/daeReader.h:241: error: template argument 3 is invalid
    /usr/local/env/graphics/openscenegraph/OpenSceneGraph-3.2.0/src/osgPlugins/dae/daeReader.h:241: error: template argument 4 is invalid
    /usr/local/env/graphics/openscenegraph/OpenSceneGraph-3.2.0/src/osgPlugins/dae/daeReader.h:243: error: 'domChannel' was not declared in this scope
    /usr/local/env/graphics/openscenegraph/OpenSceneGraph-3.2.0/src/osgPlugins/dae/daeReader.h:243: error: template argument 2 is invalid

Use "make -n" to find the failing commandline then reproduce::

    g4pb:osgPlugins blyth$ cd /usr/local/env/graphics/openscenegraph/OpenSceneGraph-3.2.0.build/src/osgPlugins/dae && /usr/bin/c++   -Dosgdb_dae_EXPORTS -DNO_BOOST -mmacosx-version-min=10.5 -ftree-vectorize -fvisibility-inlines-hidden -O3 -DNDEBUG -arch ppc -arch i386 -isysroot /Developer/SDKs/MacOSX10.5.sdk -fPIC -I/usr/local/env/graphics/openscenegraph/OpenSceneGraph-3.2.0/include -I/usr/local/env/graphics/openscenegraph/OpenSceneGraph-3.2.0.build/include -I/usr/local/include/collada-dom2.4 -I/usr/local/include/collada-dom2.4/1.4    -o CMakeFiles/osgdb_dae.dir/ReaderWriterDAE.cpp.o -c /usr/local/env/graphics/openscenegraph/OpenSceneGraph-3.2.0/src/osgPlugins/dae/ReaderWriterDAE.cpp
    /usr/local/env/graphics/openscenegraph/OpenSceneGraph-3.2.0/src/osgPlugins/dae/daeReader.h:162: error: 'domUpAxisType' does not name a type
    /usr/local/env/graphics/openscenegraph/OpenSceneGraph-3.2.0/src/osgPlugins/dae/daeReader.h:228: error: 'domFx_opaque_enum' does not name a type
    /usr/local/env/graphics/openscenegraph/OpenSceneGraph-3.2.0/src/osgPlugins/dae/daeReader.h:241: error: 'domMaterial' was not declared in this scope



Declared in 1.4/dom/domTypes.h::

    g4pb:collada-dom2.4 blyth$ pwd
    /usr/local/include/collada-dom2.4
    g4pb:collada-dom2.4 blyth$ find . -name '*.h' -exec grep -H domUpAxisType {} \;
    ...
    ./1.4/dom/domTypes.h:enum domUpAxisType {
    g4pb:collada-dom2.4 blyth$ 
    g4pb:collada-dom2.4 blyth$ 

But its inside namespace ColladaDOM141::

     012 #include <dae/daeDomTypes.h>
     013 
     014 class DAE;
     015 namespace ColladaDOM141 {
     016 
     017 typedef xsBoolean domBool;
     018 typedef xsDateTime domDateTime;
     ...
     223 enum domUpAxisType {
     224     UPAXISTYPE_X_UP,
     225     UPAXISTYPE_Y_UP,
     226     UPAXISTYPE_Z_UP,
     227     UPAXISTYPE_COUNT = 3
     228 };

::

    g4pb:collada-dom2.4 blyth$ pwd
    /usr/local/include/collada-dom2.4
    g4pb:collada-dom2.4 blyth$ find . -name '*.h' -exec grep -H using\ namespace {} \;
    ./dae/daeTypes.h:using namespace ColladaDOM150;
    ./dae/daeTypes.h:using namespace ColladaDOM141;
    ./dae/daeTypes.h:using namespace ColladaDOM150;
    ./dae/daeTypes.h:using namespace ColladaDOM141;


* http://www.dre.vanderbilt.edu/~schmidt/DOC_ROOT/ACE/docs/Symbol_Versioning.html
* :google:`openscenegraph namespace ColladaDOM141`

* https://github.com/rdiankov/collada-dom/blob/master/changelog.rst

  New Feature of Collada DOM 2.4.0

  Users can define COLLADA_DOM_USING_141 or COLLADA_DOM_USING_150 before any
  collada-dom includes in order to get an automatic "using namespace
  ColladaDOMXX".


Modify head /usr/local/env/graphics/openscenegraph/OpenSceneGraph-3.2.0/src/osgPlugins/dae/CMakeLists.txt::

    # SCB hardconfig starts
    set( CMAKE_VERBOSE_MAKEFILE ON )
    SET( COLLADA_INCLUDE_DIR     /usr/local/include/collada-dom2.4 )
    SET( COLLADA_DYNAMIC_LIBRARY /usr/local/lib/libcollada-dom2.4-dp.2.4.0.dylib )
    ADD_DEFINITIONS(-DCOLLADA_DOM_USING_141)   # for namespace new feature with colladadom 2.4.0
    # SCB hardconfig ends 

    INCLUDE_DIRECTORIES( ${COLLADA_INCLUDE_DIR} ${COLLADA_INCLUDE_DIR}/1.4)

Nope::

    Scanning dependencies of target osgdb_dae
    [ 96%] Building CXX object src/osgPlugins/dae/CMakeFiles/osgdb_dae.dir/daeReader.cpp.o
    /usr/local/include/collada-dom2.4/dae/daeTypes.h:64: error: expected namespace-name before ';' token
    /usr/local/include/collada-dom2.4/dae/daeTypes.h:64: error: '<type error>' is not a namespace
    /usr/local/include/collada-dom2.4/dae/daeTypes.h:64: error: expected namespace-name before ';' token
    /usr/local/include/collada-dom2.4/dae/daeTypes.h:64: error: '<type error>' is not a namespace
    /usr/local/env/graphics/openscenegraph/OpenSceneGraph-3.2.0/src/osgPlugins/dae/daeReader.h:162: error: 'domUpAxisType' does not name a type
    /usr/local/env/graphics/openscenegraph/OpenSceneGraph-3.2.0/src/osgPlugins/dae/daeReader.h:228: error: 'domFx_opaque_enum' does not name a type
    /usr/local/env/graphics/openscenegraph/OpenSceneGraph-3.2.0/src/osgPlugins/dae/daeReader.h:241: error: 'domMaterial' was not declared in this scope


make -n 


    cd /usr/local/env/graphics/openscenegraph/OpenSceneGraph-3.2.0.build/src/osgPlugins/dae && /usr/bin/c++   -Dosgdb_dae_EXPORTS -DCOLLADA_DOM_USING_141 -DNO_BOOST -mmacosx-version-min=10.5 -ftree-vectorize -fvisibility-inlines-hidden -O3 -DNDEBUG -arch ppc -arch i386 -isysroot /Developer/SDKs/MacOSX10.5.sdk -fPIC -I/usr/local/env/graphics/openscenegraph/OpenSceneGraph-3.2.0/include -I/usr/local/env/graphics/openscenegraph/OpenSceneGraph-3.2.0.build/include -I/usr/local/include/collada-dom2.4 -I/usr/local/include/collada-dom2.4/1.4    -o CMakeFiles/osgdb_dae.dir/ReaderWriterDAE.cpp.o -c /usr/local/env/graphics/openscenegraph/OpenSceneGraph-3.2.0/src/osgPlugins/dae/ReaderWriterDAE.cpp


/usr/local/include/collada-dom2.4/dae/daeTypes.h::

     53 
     54 #if defined(COLLADA_DOM_SUPPORT150)
     55 namespace ColladaDOM150 {}
     56 #endif
     57 #if defined(COLLADA_DOM_SUPPORT141)
     58 namespace ColladaDOM141 {}
     59 #endif
     60 
     61 #if defined(COLLADA_DOM_USING_150)
     62 using namespace ColladaDOM150;
     63 #elif defined(COLLADA_DOM_USING_141)
     64 using namespace ColladaDOM141;
     65 #elif !defined(COLLADA_DOM_NAMESPACE)

Adding the COLLADA_DOM_SUPPORT141 moves on to a different error::

    Scanning dependencies of target osgdb_dae
    [ 96%] Building CXX object src/osgPlugins/dae/CMakeFiles/osgdb_dae.dir/daeReader.cpp.o
    /usr/local/env/graphics/openscenegraph/OpenSceneGraph-3.2.0/src/osgPlugins/dae/daeReader.h:240: error: 'domGeometry' was not declared in this scope
    /usr/local/env/graphics/openscenegraph/OpenSceneGraph-3.2.0/src/osgPlugins/dae/daeReader.h:240: error: template argument 1 is invalid
    /usr/local/env/graphics/openscenegraph/OpenSceneGraph-3.2.0/src/osgPlugins/dae/daeReader.h:240: error: template argument 3 is invalid

There is a forward declaration though.

::

    239 
    240     typedef std::map<domGeometry*, osg::ref_ptr<osg::Geode> >    domGeometryGeodeMap;
    241     typedef std::map<domMaterial*, osg::ref_ptr<osg::StateSet> > domMaterialStateSetMap;
    242     typedef std::map<std::string, osg::ref_ptr<osg::StateSet> >    MaterialStateSetMap;
    243     typedef std::multimap< daeElement*, domChannel*> daeElementDomChannelMap;
    244     typedef std::map<domChannel*, osg::ref_ptr<osg::NodeCallback> > domChannelOsgAnimationUpdateCallbackMap;
    245     typedef std::map<domNode*, osg::ref_ptr<osgAnimation::Bone> > domNodeOsgBoneMap;

collada dom using namespace in a header 
-----------------------------------------

* http://stackoverflow.com/questions/5849457/using-namespace-in-c-headers


FindCOLLADA.cmake that diddles namespace definitions
------------------------------------------------------

* http://stackoverflow.com/questions/17200187/cannot-compile-collada-with-cmake-for-visual-studio




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

osg-daelink(){
  type $FUNCNAME 
  # remove -arch i386 
  cd /usr/local/env/graphics/openscenegraph/OpenSceneGraph-3.2.0.build/src/osgPlugins/dae 
  /usr/bin/c++   -mmacosx-version-min=10.5 -ftree-vectorize -fvisibility-inlines-hidden -O3 -DNDEBUG -arch ppc -isysroot /Developer/SDKs/MacOSX10.5.sdk -bundle -Wl,-headerpad_max_install_names   -o ../../../lib/osgPlugins-3.2.0/osgdb_dae.so CMakeFiles/osgdb_dae.dir/daeReader.cpp.o CMakeFiles/osgdb_dae.dir/daeRAnimations.cpp.o CMakeFiles/osgdb_dae.dir/daeRGeometry.cpp.o CMakeFiles/osgdb_dae.dir/daeRMaterials.cpp.o CMakeFiles/osgdb_dae.dir/daeRSceneObjects.cpp.o CMakeFiles/osgdb_dae.dir/daeRSkinning.cpp.o CMakeFiles/osgdb_dae.dir/daeRTransforms.cpp.o CMakeFiles/osgdb_dae.dir/daeWAnimations.cpp.o CMakeFiles/osgdb_dae.dir/daeWGeometry.cpp.o CMakeFiles/osgdb_dae.dir/daeWMaterials.cpp.o CMakeFiles/osgdb_dae.dir/daeWriter.cpp.o CMakeFiles/osgdb_dae.dir/daeWSceneObjects.cpp.o CMakeFiles/osgdb_dae.dir/daeWTransforms.cpp.o CMakeFiles/osgdb_dae.dir/domSourceReader.cpp.o CMakeFiles/osgdb_dae.dir/ReaderWriterDAE.cpp.o ../../../lib/libOpenThreads.3.2.0.dylib ../../../lib/libosg.3.2.0.dylib ../../../lib/libosgDB.3.2.0.dylib ../../../lib/libosgUtil.3.2.0.dylib ../../../lib/libosgSim.3.2.0.dylib ../../../lib/libosgAnimation.3.2.0.dylib /usr/local/lib/libcollada-dom2.4-dp.2.4.0.dylib /opt/local/lib/libboost_filesystem.dylib ../../../lib/libosgViewer.3.2.0.dylib ../../../lib/libosgText.3.2.0.dylib ../../../lib/libosgGA.3.2.0.dylib ../../../lib/libosgDB.3.2.0.dylib -framework Carbon /usr/lib/libz.dylib ../../../lib/libosgUtil.3.2.0.dylib ../../../lib/libosg.3.2.0.dylib ../../../lib/libOpenThreads.3.2.0.dylib -lpthread /usr/lib/libm.dylib /usr/lib/libdl.dylib -framework Cocoa -framework OpenGL  

   sudo make install 
}



