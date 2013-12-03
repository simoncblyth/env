Qt Mac Deployment
===================

This is somewhat academic from my ancient PPC machine as all 
the dylibs are only PPC anyhow and it would be difficult 
to find a machine to deploy to::

    simon:Frameworks blyth$ file libcrypto.1.0.0.dylib
    libcrypto.1.0.0.dylib: Mach-O dynamically linked shared library ppc
    simon:Frameworks blyth$ file libdbus-1.3.dylib
    libdbus-1.3.dylib: Mach-O dynamically linked shared library ppc


* http://qt-project.org/doc/qt-4.8/deployment-mac.html
* http://qt-project.org/doc/qt-4.8/deployment-mac.html#macdeploy

* https://qt.gitorious.org/qt/qttools/source/42076738c5c73b2bc61d021fa02a06e46ef46bce:src/macdeployqt/macdeployqt/main.cpp
* https://qt.gitorious.org/qt/qttools/source/42076738c5c73b2bc61d021fa02a06e46ef46bce:src/macdeployqt/shared/shared.h
* https://qt.gitorious.org/qt/qttools/source/42076738c5c73b2bc61d021fa02a06e46ef46bce:src/macdeployqt/shared/shared.cpp


To switch off the bundling::

   CONFIG-=app_bundle


macdeployqt tool
------------------

::

    simon:meshlab blyth$ which macdeployqt
    /opt/local/bin/macdeployqt
    simon:meshlab blyth$ macdeployqt -h
    Usage: macdeployqt app-bundle [options] 
     
    Options: 
       -verbose=<0-3>     : 0 = no output, 1 = error/warning (default), 2 = normal, 3 = debug 
       -no-plugins        : Skip plugin deployment 
       -dmg               : Create a .dmg disk image 
       -no-strip          : Don't run 'strip' on the binaries 
       -use-debug-libs    : Deploy with debug versions of frameworks and plugins (implies -no-strip) 
       -executable=<path> : Let the given executable use the deployed frameworks too 
     
    macdeployqt takes an application bundle as input and makes it 
    self-contained by copying in the Qt frameworks and plugins that 
    the application uses. 
     
    Plugins related to a framework are copied in with the 
    framework. The accessibilty, image formats, and text codec 
    plugins are always copied, unless "-no-plugins" is specified. 
     
    See the "Deploying an Application on Qt/Mac" topic in the 
    documentation for more information about deployment on Mac OS X. 


::

    simon:src blyth$ macdeployqt distrib/meshlab.app
    ERROR: "install_name_tool: can't open input file: distrib/meshlab.app/Contents/Frameworks//libssl.1.0.0.dylib for writing (Permission denied)
    install_name_tool: can't lseek to offset: 0 in file: distrib/meshlab.app/Contents/Frameworks//libssl.1.0.0.dylib for writing (Bad file descriptor)
    install_name_tool: can't write new headers in file: distrib/meshlab.app/Contents/Frameworks//libssl.1.0.0.dylib (Bad file descriptor)
    install_name_tool: can't close written on input file: distrib/meshlab.app/Contents/Frameworks//libssl.1.0.0.dylib (Bad file descriptor)
    " 
    ERROR: "" 
    ERROR: "install_name_tool: can't open input file: distrib/meshlab.app/Contents/Frameworks//libssl.1.0.0.dylib for writing (Permission denied)
    install_name_tool: can't lseek to offset: 0 in file: distrib/meshlab.app/Contents/Frameworks//libssl.1.0.0.dylib for writing (Bad file descriptor)
    install_name_tool: can't write new headers in file: distrib/meshlab.app/Contents/Frameworks//libssl.1.0.0.dylib (Bad file descriptor)
    install_name_tool: can't close written on input file: distrib/meshlab.app/Contents/Frameworks//libssl.1.0.0.dylib (Bad file descriptor)
    " 
    ERROR: "" 
    ERROR: "install_name_tool: can't open input file: distrib/meshlab.app/Contents/Frameworks//libssl.1.0.0.dylib for writing (Permission denied)
    install_name_tool: can't lseek to offset: 0 in file: distrib/meshlab.app/Contents/Frameworks//libssl.1.0.0.dylib for writing (Bad file descriptor)
    install_name_tool: can't write new headers in file: distrib/meshlab.app/Contents/Frameworks//libssl.1.0.0.dylib (Bad file descriptor)
    install_name_tool: can't close written on input file: distrib/meshlab.app/Contents/Frameworks//libssl.1.0.0.dylib (Bad file descriptor)
    " 
    ERROR: "" 
    ERROR: "install_name_tool: can't open input file: distrib/meshlab.app/Contents/Frameworks//libcrypto.1.0.0.dylib for writing (Permission denied)
    install_name_tool: can't lseek to offset: 0 in file: distrib/meshlab.app/Contents/Frameworks//libcrypto.1.0.0.dylib for writing (Bad file descriptor)
    install_name_tool: can't write new headers in file: distrib/meshlab.app/Contents/Frameworks//libcrypto.1.0.0.dylib (Bad file descriptor)
    install_name_tool: can't close written on input file: distrib/meshlab.app/Contents/Frameworks//libcrypto.1.0.0.dylib (Bad file descriptor)
    " 
    ERROR: "" 
    ERROR: "install_name_tool: can't open input file: distrib/meshlab.app/Contents/Frameworks//libcrypto.1.0.0.dylib for writing (Permission denied)
    install_name_tool: can't lseek to offset: 0 in file: distrib/meshlab.app/Contents/Frameworks//libcrypto.1.0.0.dylib for writing (Bad file descriptor)
    install_name_tool: can't write new headers in file: distrib/meshlab.app/Contents/Frameworks//libcrypto.1.0.0.dylib (Bad file descriptor)
    install_name_tool: can't close written on input file: distrib/meshlab.app/Contents/Frameworks//libcrypto.1.0.0.dylib (Bad file descriptor)
    " 
    ERROR: "" 
    simon:src blyth$ 



::

    simon:src blyth$ macdeployqt distrib/meshlab.app -verbose=3
    Log: Using otool: 
    Log:  inspecting "distrib/meshlab.app/Contents/MacOS/meshlab" 
    Log: Using otool: 
    Log:  inspecting "distrib/meshlab.app/Contents/Frameworks/libcrypto.1.0.0.dylib" 
    Log: Adding framework: 
    ...


Try changing permissions inplace and rerunning::

    simon:src blyth$ chmod u+w distrib/meshlab.app/Contents/Frameworks/libcrypto.1.0.0.dylib 
    simon:src blyth$ chmod u+w distrib/meshlab.app/Contents/Frameworks/libssl.1.0.0.dylib 
    simon:src blyth$ macdeployqt distrib/meshlab.app -verbose=3
    ...
    Log: Using strip: 
    Log:  stripped "distrib/meshlab.app/Contents/Frameworks//libcrypto.1.0.0.dylib" 
    Log: Using install_name_tool: 
    Log:  change identification in "distrib/meshlab.app/Contents/Frameworks//libcrypto.1.0.0.dylib" 
    Log:  to "@executable_path/../Frameworks/libcrypto.1.0.0.dylib" 
    Log: Using otool: 
    Log:  inspecting "distrib/meshlab.app/Contents/Frameworks//libcrypto.1.0.0.dylib" 
    Log: 
    Log: Deploying plugins from "/opt/local/share/qt4/plugins" 
    Log: Deploying plugins from "/opt/local/share/qt4/plugins/accessible" 
    ERROR: file copy failed from "/opt/local/share/qt4/plugins/accessible/libqtaccessiblewidgets.dylib" 
    ERROR:  to "distrib/meshlab.app/Contents/PlugIns/accessible/libqtaccessiblewidgets.dylib" 
    Log: Deploying plugins from "/opt/local/share/qt4/plugins/bearer" 
    Log: Deploying plugins from "/opt/local/share/qt4/plugins/codecs" 
    ERROR: file copy failed from "/opt/local/share/qt4/plugins/codecs/libqcncodecs.dylib" 
    ERROR:  to "distrib/meshlab.app/Contents/PlugIns/codecs/libqcncodecs.dylib" 
    ERROR: file copy failed from "/opt/local/share/qt4/plugins/codecs/libqjpcodecs.dylib" 
    ERROR:  to "distrib/meshlab.app/Contents/PlugIns/codecs/libqjpcodecs.dylib" 
    ERROR: file copy failed from "/opt/local/share/qt4/plugins/codecs/libqkrcodecs.dylib" 
    ERROR:  to "distrib/meshlab.app/Contents/PlugIns/codecs/libqkrcodecs.dylib" 
    ERROR: file copy failed from "/opt/local/share/qt4/plugins/codecs/libqtwcodecs.dylib" 
    ERROR:  to "distrib/meshlab.app/Contents/PlugIns/codecs/libqtwcodecs.dylib" 
    Log: Deploying plugins from "/opt/local/share/qt4/plugins/designer" 
    Log: Deploying plugins from "/opt/local/share/qt4/plugins/graphicssystems" 
    Log: Deploying plugins from "/opt/local/share/qt4/plugins/iconengines" 
    Log: Deploying plugins from "/opt/local/share/qt4/plugins/imageformats" 
    ERROR: file copy failed from "/opt/local/share/qt4/plugins/imageformats/libqgif.dylib" 
    ERROR:  to "distrib/meshlab.app/Contents/PlugIns/imageformats/libqgif.dylib" 
    ERROR: file copy failed from "/opt/local/share/qt4/plugins/imageformats/libqico.dylib" 
    ERROR:  to "distrib/meshlab.app/Contents/PlugIns/imageformats/libqico.dylib" 
    ERROR: file copy failed from "/opt/local/share/qt4/plugins/imageformats/libqjpeg.dylib" 
    ERROR:  to "distrib/meshlab.app/Contents/PlugIns/imageformats/libqjpeg.dylib" 
    ERROR: file copy failed from "/opt/local/share/qt4/plugins/imageformats/libqmng.dylib" 
    ERROR:  to "distrib/meshlab.app/Contents/PlugIns/imageformats/libqmng.dylib" 
    ERROR: file copy failed from "/opt/local/share/qt4/plugins/imageformats/libqtga.dylib" 
    ERROR:  to "distrib/meshlab.app/Contents/PlugIns/imageformats/libqtga.dylib" 
    ERROR: file copy failed from "/opt/local/share/qt4/plugins/imageformats/libqtiff.dylib" 
    ERROR:  to "distrib/meshlab.app/Contents/PlugIns/imageformats/libqtiff.dylib" 
    Log: Deploying plugins from "/opt/local/share/qt4/plugins/qmltooling" 
    Log: Deploying plugins from "/opt/local/share/qt4/plugins/script" 
    WARNING: 
    WARNING: "distrib/meshlab.app/Contents/Resources/qt.conf" already exists, will not overwrite. 
    WARNING: To make sure the plugins are loaded from the correct location, 
    WARNING: please make sure qt.conf contains the following lines: 
    WARNING: [Paths] 
    WARNING:   Plugins = PlugIns 


::

    simon:src blyth$ macdeployqt distrib/meshlab.app           
    WARNING: 
    WARNING: Could not find any external Qt frameworks to deploy in "distrib/meshlab.app" 
    WARNING: Perhaps macdeployqt was already used on "distrib/meshlab.app" ? 
    WARNING: If so, you will need to rebuild "distrib/meshlab.app" before trying again. 
    ERROR: file copy failed from "/opt/local/share/qt4/plugins/accessible/libqtaccessiblewidgets.dylib" 
    ERROR:  to "distrib/meshlab.app/Contents/PlugIns/accessible/libqtaccessiblewidgets.dylib" 
    ERROR: file copy failed from "/opt/local/share/qt4/plugins/codecs/libqcncodecs.dylib" 
    ERROR:  to "distrib/meshlab.app/Contents/PlugIns/codecs/libqcncodecs.dylib" 
    ERROR: file copy failed from "/opt/local/share/qt4/plugins/codecs/libqjpcodecs.dylib" 
    ERROR:  to "distrib/meshlab.app/Contents/PlugIns/codecs/libqjpcodecs.dylib" 
    ERROR: file copy failed from "/opt/local/share/qt4/plugins/codecs/libqkrcodecs.dylib" 
    ERROR:  to "distrib/meshlab.app/Contents/PlugIns/codecs/libqkrcodecs.dylib" 
    ERROR: file copy failed from "/opt/local/share/qt4/plugins/codecs/libqtwcodecs.dylib" 
    ERROR:  to "distrib/meshlab.app/Contents/PlugIns/codecs/libqtwcodecs.dylib" 
    ERROR: file copy failed from "/opt/local/share/qt4/plugins/imageformats/libqgif.dylib" 
    ERROR:  to "distrib/meshlab.app/Contents/PlugIns/imageformats/libqgif.dylib" 
    ERROR: file copy failed from "/opt/local/share/qt4/plugins/imageformats/libqico.dylib" 
    ERROR:  to "distrib/meshlab.app/Contents/PlugIns/imageformats/libqico.dylib" 
    ERROR: file copy failed from "/opt/local/share/qt4/plugins/imageformats/libqjpeg.dylib" 
    ERROR:  to "distrib/meshlab.app/Contents/PlugIns/imageformats/libqjpeg.dylib" 
    ERROR: file copy failed from "/opt/local/share/qt4/plugins/imageformats/libqmng.dylib" 
    ERROR:  to "distrib/meshlab.app/Contents/PlugIns/imageformats/libqmng.dylib" 
    ERROR: file copy failed from "/opt/local/share/qt4/plugins/imageformats/libqtga.dylib" 
    ERROR:  to "distrib/meshlab.app/Contents/PlugIns/imageformats/libqtga.dylib" 
    ERROR: file copy failed from "/opt/local/share/qt4/plugins/imageformats/libqtiff.dylib" 
    ERROR:  to "distrib/meshlab.app/Contents/PlugIns/imageformats/libqtiff.dylib" 
    WARNING: 
    WARNING: "distrib/meshlab.app/Contents/Resources/qt.conf" already exists, will not overwrite. 
    WARNING: To make sure the plugins are loaded from the correct location, 
    WARNING: please make sure qt.conf contains the following lines: 
    WARNING: [Paths] 
    WARNING:   Plugins = PlugIns 
    simon:src blyth$ 
    simon:src blyth$ cat distrib/meshlab.app/Contents/Resources/qt.conf
    [Paths]
    Plugins = PlugIns



::

    simon:src blyth$ du -hs distrib/meshlab.app/Contents/*
     29M    distrib/meshlab.app/Contents/Frameworks
    8.2M    distrib/meshlab.app/Contents/MacOS
    4.0K    distrib/meshlab.app/Contents/PkgInfo
    1.2M    distrib/meshlab.app/Contents/PlugIns
    4.0K    distrib/meshlab.app/Contents/Resources




* http://qt-project.org/faq/answer/errors_using_macdeployqt_caused_by_old_version_of_xcode
* http://stackoverflow.com/questions/10560811/why-does-macdeployqt-give-the-error-cant-open-input-file
* :google:`macdeployqt libcrypto libssl`


* http://code.google.com/p/companion9x/wiki/MacBuild







::

    simon:src blyth$ l  distrib/meshlab.app/Contents/Frameworks/
    total 7448
    -rwxr-xr-x  1 blyth  wheel   197788  3 Dec 12:46 libjpeg.9.dylib
    -rwxr-xr-x  1 blyth  wheel   132192  3 Dec 12:46 liblzma.5.dylib
    -rwxr-xr-x  1 blyth  wheel    81876  3 Dec 12:46 libz.1.dylib
    -rwxr-xr-x  1 blyth  wheel   414792  3 Dec 12:46 libtiff.5.dylib
    -rwxr-xr-x  1 blyth  wheel   183508  3 Dec 12:46 liblcms.1.dylib
    -rwxr-xr-x  1 blyth  wheel   380084  3 Dec 12:45 libmng.1.dylib
    -r-xr-xr-x  1 blyth  wheel  1614632  3 Dec 12:45 libcrypto.1.0.0.dylib    # missing w permission
    -rwxr-xr-x  1 blyth  wheel   146972  3 Dec 12:45 libpng15.15.dylib
    -rwxr-xr-x  1 blyth  wheel   244456  3 Dec 12:45 libdbus-1.3.dylib
    -r-xr-xr-x  1 blyth  wheel   399212  3 Dec 12:45 libssl.1.0.0.dylib       # missing w permission 
    drwxr-xr-x  4 blyth  wheel      136  3 Dec 12:45 QtGui.framework
    drwxr-xr-x  4 blyth  wheel      136  3 Dec 12:45 QtNetwork.framework
    drwxr-xr-x  4 blyth  wheel      136  3 Dec 12:45 QtOpenGL.framework
    drwxr-xr-x  4 blyth  wheel      136  3 Dec 12:45 QtXmlPatterns.framework
    drwxr-xr-x  4 blyth  wheel      136  3 Dec 12:45 QtScript.framework
    drwxr-xr-x  4 blyth  wheel      136  3 Dec 12:45 QtCore.framework
    drwxr-xr-x  4 blyth  wheel      136  3 Dec 12:45 QtDBus.framework
    drwxr-xr-x  4 blyth  wheel      136  3 Dec 12:45 QtXml.framework
    simon:src blyth$ 
    simon:src blyth$ date
    Tue  3 Dec 2013 12:48:29 CST


::

    simon:src blyth$ otool -L distrib/meshlab.app/Contents/MacOS/meshlab 
    distrib/meshlab.app/Contents/MacOS/meshlab:
            @executable_path/libcommon.1.dylib (compatibility version 1.0.0, current version 1.0.0)
            @executable_path/../Frameworks/QtDBus.framework/Versions/4/QtDBus (compatibility version 4.8.0, current version 4.8.5)
            @executable_path/../Frameworks/QtXml.framework/Versions/4/QtXml (compatibility version 4.8.0, current version 4.8.5)
            @executable_path/../Frameworks/QtCore.framework/Versions/4/QtCore (compatibility version 4.8.0, current version 4.8.5)
            @executable_path/../Frameworks/QtScript.framework/Versions/4/QtScript (compatibility version 4.8.0, current version 4.8.5)
            @executable_path/../Frameworks/QtXmlPatterns.framework/Versions/4/QtXmlPatterns (compatibility version 4.8.0, current version 4.8.5)
            @executable_path/../Frameworks/QtNetwork.framework/Versions/4/QtNetwork (compatibility version 4.8.0, current version 4.8.5)
            @executable_path/../Frameworks/QtOpenGL.framework/Versions/4/QtOpenGL (compatibility version 4.8.0, current version 4.8.5)
            @executable_path/../Frameworks/QtGui.framework/Versions/4/QtGui (compatibility version 4.8.0, current version 4.8.5)
            /System/Library/Frameworks/OpenGL.framework/Versions/A/OpenGL (compatibility version 1.0.0, current version 1.0.0)
            /System/Library/Frameworks/AGL.framework/Versions/A/AGL (compatibility version 1.0.0, current version 1.0.0)
            /usr/lib/libstdc++.6.dylib (compatibility version 7.0.0, current version 7.4.0)
            /usr/lib/libgcc_s.1.dylib (compatibility version 1.0.0, current version 1.0.0)
            /usr/lib/libSystem.B.dylib (compatibility version 1.0.0, current version 111.1.7)
    simon:src blyth$ 

::

    simon:src blyth$ otool -L distrib/meshlab.app/Contents/MacOS/../Frameworks/QtDBus.framework/Versions/4/QtDBus
    distrib/meshlab.app/Contents/MacOS/../Frameworks/QtDBus.framework/Versions/4/QtDBus:
            @executable_path/../Frameworks/QtDBus.framework/Versions/4/QtDBus (compatibility version 4.8.0, current version 4.8.5)
            @executable_path/../Frameworks/QtXml.framework/Versions/4/QtXml (compatibility version 4.8.0, current version 4.8.5)
            @executable_path/../Frameworks/QtCore.framework/Versions/4/QtCore (compatibility version 4.8.0, current version 4.8.5)
            @executable_path/../Frameworks/libdbus-1.3.dylib (compatibility version 11.0.0, current version 11.4.0)
            /usr/lib/libstdc++.6.dylib (compatibility version 7.0.0, current version 7.4.0)
            /usr/lib/libgcc_s.1.dylib (compatibility version 1.0.0, current version 1.0.0)
            /usr/lib/libSystem.B.dylib (compatibility version 1.0.0, current version 111.0.0)



Macports dependency remains due to permissions issue::

    simon:src blyth$ otool -L distrib/meshlab.app/Contents/Frameworks/libcrypto.1.0.0.dylib
    distrib/meshlab.app/Contents/Frameworks/libcrypto.1.0.0.dylib:
            /opt/local/lib/libcrypto.1.0.0.dylib (compatibility version 1.0.0, current version 1.0.0)
            /opt/local/lib/libz.1.dylib (compatibility version 1.0.0, current version 1.2.8)
            /usr/lib/libgcc_s.1.dylib (compatibility version 1.0.0, current version 1.0.0)
            /usr/lib/libSystem.B.dylib (compatibility version 1.0.0, current version 111.1.7)
    simon:src blyth$ 

    simon:src blyth$ otool -L distrib/meshlab.app/Contents/Frameworks/libssl.1.0.0.dylib   
    distrib/meshlab.app/Contents/Frameworks/libssl.1.0.0.dylib:
            /opt/local/lib/libssl.1.0.0.dylib (compatibility version 1.0.0, current version 1.0.0)
            /opt/local/lib/libcrypto.1.0.0.dylib (compatibility version 1.0.0, current version 1.0.0)
            /opt/local/lib/libz.1.dylib (compatibility version 1.0.0, current version 1.2.8)
            /usr/lib/libgcc_s.1.dylib (compatibility version 1.0.0, current version 1.0.0)
            /usr/lib/libSystem.B.dylib (compatibility version 1.0.0, current version 111.1.7)

Problems come from QtNetwork::

    simon:src blyth$ otool -L distrib/meshlab.app/Contents/MacOS/../Frameworks/QtNetwork.framework/Versions/4/QtNetwork 
    distrib/meshlab.app/Contents/MacOS/../Frameworks/QtNetwork.framework/Versions/4/QtNetwork:
            @executable_path/../Frameworks/QtNetwork.framework/Versions/4/QtNetwork (compatibility version 4.8.0, current version 4.8.5)
            @executable_path/../Frameworks/QtCore.framework/Versions/4/QtCore (compatibility version 4.8.0, current version 4.8.5)
            @executable_path/../Frameworks/libz.1.dylib (compatibility version 1.0.0, current version 1.2.8)
            /System/Library/Frameworks/SystemConfiguration.framework/Versions/A/SystemConfiguration (compatibility version 1.0.0, current version 204.0.0)
            /System/Library/Frameworks/CoreFoundation.framework/Versions/A/CoreFoundation (compatibility version 150.0.0, current version 476.0.0)
            /System/Library/Frameworks/CoreServices.framework/Versions/A/CoreServices (compatibility version 1.0.0, current version 32.0.0)
            @executable_path/../Frameworks/libssl.1.0.0.dylib (compatibility version 1.0.0, current version 1.0.0)
            @executable_path/../Frameworks/libcrypto.1.0.0.dylib (compatibility version 1.0.0, current version 1.0.0)
            /usr/lib/libstdc++.6.dylib (compatibility version 7.0.0, current version 7.4.0)
            /usr/lib/libgcc_s.1.dylib (compatibility version 1.0.0, current version 1.0.0)
            /usr/lib/libSystem.B.dylib (compatibility version 1.0.0, current version 111.0.0)
    simon:src blyth$ 


* https://bugreports.qt-project.org/browse/QTBUG-33113

Possible workaround is to use ``-no-opensll`` option with Qt (qmake presumably).




