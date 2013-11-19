# === func-gen- : graphics/mesh/meshlab fgp graphics/mesh/meshlab.bash fgn meshlab fgh graphics/mesh
meshlab-src(){      echo graphics/meshlab/meshlab.bash ; }
meshlab-source(){   echo ${BASH_SOURCE:-$(env-home)/$(meshlab-src)} ; }
meshlab-vi(){       vi $(meshlab-source) ; }
meshlab-usage(){ cat << EOU
MESHLAB
========

.. contents:: :local:


Overview
---------

Qt based GUI for mesh viewing/manipulation

* http://meshlab.sourceforge.net/
* http://vcg.isti.cnr.it/~cignoni/newvcglib/html/
* http://sourceforge.net/apps/mediawiki/meshlab/index.php?title=Main_Page


    MeshLab is a advanced mesh processing system, for the automatic and user
    assisted editing, cleaning, filtering converting and rendering of large
    unstructured 3D triangular meshes. MeshLab is actively developed by the a small
    group of people at the Visual Computing Lab at the ISTI - CNR institute, a
    large group of university students and some great developers from the rest of
    the world. For the basic mesh processing tasks and for the internal data
    structures the system relies on the GPL VCG library.

VCG
----

    The Visualization and Computer Graphics Library (VCG for short) is a open
    source portable C++ templated library for manipulation, processing and
    displaying with OpenGL of triangle and tetrahedral meshes.

    The library, composed by more than 100k lines of code, is released under the
    GPL license, and it is the base of most of the software tools of the Visual
    Computing Lab of the Italian National Research Council Institute ISTI
    (http://vcg.isti.cnr.it), like metro and MeshLab.



Observations
-------------

#. supports VRML/X3D/STL import and export 
#. documentation is sparse
#. billed as for handling *unstrucutured* meshes : does that mean it will flatten the scene graph ?


GUI Navigation with Intel Mac Laptop
-------------------------------------

* http://sourceforge.net/apps/mediawiki/meshlab/index.php?title=Interacting_with_the_mesh


esc
             toggle camera/edit mode
shift-cmd-H
             return to home position 
double click
             change center of rotation
one finger drag
             move viewpoint about rotation position 
cmd-drag
             pan (a big P appears)


two-finger (wheel) 
             dolly forward/backward 
             (mis-named zoom, zoom really means changing FOV)
shift+two-finger (shift wheel)
             change camera field of view, 
             keep reducing it down to 5 degrees to switch to Orthographic
cmd+two-finger (ctrl wheel)
             move near clipping plane
cmd+shift+two-finger (ctrl-shift wheel)
             move far clipping plane


shift+cmd+one-finger
             change light direction         

opt/alt-return
             toggle fullscreen mode


Tip for usage on underpowered machine
---------------------------------------

Switch to point rendering (rather than wireframe or solid) when navigating 
to where you want to be. 

Point rendering manages ~4 fps whereas the others are difficult 
to use at 0.3 fps.  Then switch to nicer rendering when in position.





Links
------

Code

* http://svn.code.sf.net/p/meshlab/code/trunk/meshlab/
* http://svn.code.sf.net/p/vcg/code/trunk/vcglib/

* http://svn.code.sf.net/p/vcg/code/trunk/vcglib/wrap/io_trimesh/

  * huh, no vrml import contrary to frontpage 

Forum

* http://sourceforge.net/p/meshlab/discussion/499533

Other

* http://meshlabstuff.blogspot.tw/ 
* https://itunes.apple.com/us/app/meshlab-for-ios/id451944013
* http://www.meshpad.org/


iOS source ? (hmm GPL)
~~~~~~~~~~~~~~~~~~~~~~~

Quote from http://sourceforge.net/p/meshlab/discussion/499533/thread/836b4da1/

    The source code of MeshLab for iOS and MeshLab for Android is not available for
    the general public. We can provide customized version of it, or license the
    viewing component under a commercial agreement.


Pre-requisites
---------------

* http://meshlab.sourceforge.net/wiki/index.php/Compiling

* Qt 4.8 (note that Qt 4.7 are required, Qt versions < 4.7 could not compile).
  Current version of MeshLab compiles well against Qt 4.7.4.



Download
----------

::

   mv ~/Downloads/MeshLabSrc_AllInc_v132.tar .   ## WARNING : exploding tarball

Tarball explodes to create the below::

    simon:meshlab blyth$ l
    total 94552
    drwxr-xr-x@ 7 blyth  wheel       238 22 Aug 13:16 vcglib
    drwxr-xr-x@ 3 blyth  wheel       102 22 Aug 13:15 meshlab
    -rw-r--r--@ 1 blyth  staff  48404480 22 Aug 13:07 MeshLabSrc_AllInc_v132.tar
    -rw-r--r--@ 1 blyth  wheel       150  3 Aug  2012 how_to_compile.txt
    simon:meshlab blyth$ pwd
    /usr/local/env/graphics/mesh/graphics/meshlab


G Build
---------

Following

* http://sourceforge.net/apps/mediawiki/meshlab/index.php?title=Compiling
* http://www.sentex.net/~mwandel/jhead/  

  * jpeg metadata extractor ? why is that ``absolutely required``

::

    simon:meshlab blyth$ qmake -v    
    QMake version 2.01a
    Using Qt version 4.8.5 in /opt/local/lib
    simon:meshlab blyth$ 
    simon:meshlab blyth$ pwd
    /usr/local/env/graphics/meshlab
    simon:meshlab blyth$ find . -name '*.pro'    # look for qmake project files
    ./meshlab/src/common/common.pro
    ./meshlab/src/external/ann_1.1.1/ANN.pro
    ./meshlab/src/external/bzip2-1.0.5/bzip2-1.0.5.pro
    ./meshlab/src/external/external.pro
    ./meshlab/src/external/jhead-2.95/jhead-2.95.pro
    ./meshlab/src/external/levmar-2.3/levmar-2.3.pro
    ./meshlab/src/external/lib3ds-1.3.0/lib3ds/lib3ds.pro
    ...
    simon:meshlab blyth$ find . -name '*.pro' | wc -l 
         179

    simon:meshlab blyth$ find . -name external.pro
    ./meshlab/src/external/external.pro
    simon:meshlab blyth$ vi meshlab/src/external/external.pro
    simon:meshlab blyth$ cat  meshlab/src/external/external.pro
    config += debug_and_release

    TEMPLATE      = subdirs

    SUBDIRS       = lib3ds-1.3.0/lib3ds \
                    bzip2-1.0.5/bzip2-1.0.5.pro \
                    muparser_v132/src \
                    levmar-2.3/levmar-2.3.pro \
                    structuresynth/structuresynth.pro \
                                    OpenCTM-1.0.3/openctm.pro \
                    jhead-2.95/jhead-2.95.pro
#                openkinect/openkinect.pro
    simon:meshlab blyth$ 

    

EXTERNALS
-----------

::

    simon:external blyth$ qmake -recursive external.pro
    ...
    simon:external blyth$ find . -name Makefile
    ./lib3ds-1.3.0/lib3ds/Makefile
    ./Makefile
    ./muparser_v132/src/Makefile

* http://svn.code.sf.net/p/meshlab/code/trunk/meshlab/src/external/lib/macx32/

  * hmm static libs in SVN ?

After ``make``::

    simon:external blyth$ l lib/macx/
    total 3616
    -rw-r--r--  1 blyth  wheel   75976 18 Nov 11:48 libjhead.a
    -rw-r--r--  1 blyth  wheel   82272 18 Nov 11:48 libopenctm.a
    -rw-r--r--  1 blyth  wheel  755532 18 Nov 11:48 libssynth.a
    -rw-r--r--  1 blyth  wheel   60528 18 Nov 11:39 liblevmar.a
    -rw-r--r--  1 blyth  wheel  584540 18 Nov 11:39 libmuparser.a
    -rw-r--r--  1 blyth  wheel  131748 18 Nov 11:37 libbz2.a
    -rw-r--r--  1 blyth  wheel  146872 18 Nov 11:37 lib3ds.a


Meshlab mini
-------------

qmake trying to use clang++
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

And it refuses to be overridden on cmdline::

    simon:src blyth$ qmake -recursive meshlab_mini.pro
    simon:src blyth$ make    
    cd common/ && make -f Makefile 
    ...
    make[1]: clang++: Command not found

qmake generated src/common/Makefile::

     11 CC            = /usr/bin/gcc-4.2
     12 CXX           = clang++


qmake CXX sticks to clang++ despite spec settings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Default::

     6 # Command: /opt/local/bin/qmake -o Makefile common.pro
     ... 
     11 CC            = /usr/bin/gcc-4.2
     12 CXX           = clang++


``qmake -spec macx-g++40 ``::

      6 # Command: /opt/local/bin/qmake -spec /opt/local/share/qt4/mkspecs/macx-g++40 -o Makefile common.pro
      ...
      11 CC            = gcc-4.0
      12 CXX           = clang++

See details in *qt4-*.


qmake workaround, inplace edit the Makefiles
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The macports Portfile is a mess, so just kludge it::

    simon:src blyth$ qmake -recursive meshlab_mini.pro
    simon:src blyth$ find . -name Makefile -exec perl -pi -e 's,clang,g,g' {} \;     ## now in qt4-kludge
    simon:src blyth$ open distrib/meshlab.app

launch crash
---------------

From the report::

    Date/Time:       2013-11-18 13:38:02.081 +0800
    OS Version:      Mac OS X 10.5.8 (9L31a)
    Report Version:  6
    Anonymous UUID:  0AEE87B7-11A3-4A84-B851-87CA48233147

    Exception Type:  EXC_CRASH (SIGABRT)
    Exception Codes: 0x0000000000000000, 0x0000000000000000
    Crashed Thread:  0

    Thread 0 Crashed:
    0   libSystem.B.dylib               0x957659f0 __kill + 12
    1   libSystem.B.dylib               0x95800bf8 abort + 84
    2   libstdc++.6.dylib               0x91c4de24 __gnu_cxx::__verbose_terminate_handler() + 400
    3   libstdc++.6.dylib               0x91c4b940 __gxx_personality_v0 + 1240
    4   libstdc++.6.dylib               0x91c4b9a4 std::terminate() + 68
    5   libstdc++.6.dylib               0x91c4bbe4 __cxa_throw + 124
    6   libcommon.1.dylib               0x00244fa4 PluginManager::loadXMLPlugin(QString const&) + 3380
    7   libcommon.1.dylib               0x00247090 PluginManager::loadPlugins(RichParameterSet&) + 2496
    8   meshlab                         0x000161b8 MainWindow::MainWindow() + 920
    9   meshlab                         0x00006b98 main + 920
    10  meshlab                         0x00005a00 start + 64


::

    simon:MacOS blyth$ pwd
    /usr/local/env/graphics/meshlab/meshlab/src/distrib/meshlab.app/Contents/MacOS
    simon:MacOS blyth$ otool -L meshlab
    meshlab:
            @executable_path/libcommon.1.dylib (compatibility version 1.0.0, current version 1.0.0)
            /opt/local/Library/Frameworks/QtScript.framework/Versions/4/QtScript (compatibility version 4.8.0, current version 4.8.5)
            /opt/local/Library/Frameworks/QtCore.framework/Versions/4/QtCore (compatibility version 4.8.0, current version 4.8.5)
            /opt/local/Library/Frameworks/QtXmlPatterns.framework/Versions/4/QtXmlPatterns (compatibility version 4.8.0, current version 4.8.5)
            /opt/local/Library/Frameworks/QtNetwork.framework/Versions/4/QtNetwork (compatibility version 4.8.0, current version 4.8.5)
            /opt/local/Library/Frameworks/QtXml.framework/Versions/4/QtXml (compatibility version 4.8.0, current version 4.8.5)
            /opt/local/Library/Frameworks/QtOpenGL.framework/Versions/4/QtOpenGL (compatibility version 4.8.0, current version 4.8.5)
            /opt/local/Library/Frameworks/QtGui.framework/Versions/4/QtGui (compatibility version 4.8.0, current version 4.8.5)
            /System/Library/Frameworks/OpenGL.framework/Versions/A/OpenGL (compatibility version 1.0.0, current version 1.0.0)
            /System/Library/Frameworks/AGL.framework/Versions/A/AGL (compatibility version 1.0.0, current version 1.0.0)
            /usr/lib/libstdc++.6.dylib (compatibility version 7.0.0, current version 7.4.0)
            /usr/lib/libgcc_s.1.dylib (compatibility version 1.0.0, current version 1.0.0)
            /usr/lib/libSystem.B.dylib (compatibility version 1.0.0, current version 111.1.7)
    simon:MacOS blyth$ 
    simon:MacOS blyth$ gdb meshlab 
    GNU gdb 6.3.50-20050815 (Apple version gdb-967) (Tue Jul 14 02:15:14 UTC 2009)
    Copyright 2004 Free Software Foundation, Inc.
    GDB is free software, covered by the GNU General Public License, and you are
    welcome to change it and/or distribute copies of it under certain conditions.
    Type "show copying" to see the conditions.
    There is absolutely no warranty for GDB.  Type "show warranty" for details.
    This GDB was configured as "powerpc-apple-darwin"...Reading symbols for shared libraries .............. done

    (gdb) r
    Starting program: /usr/local/env/graphics/meshlab/meshlab/src/distrib/meshlab.app/Contents/MacOS/meshlab 
    ...
    Reading symbols for shared libraries ... done
    The base dir is /usr/local/env/graphics/meshlab/meshlab/src/distrib
    The base dir is /usr/local/env/graphics/meshlab/meshlab/src/distrib
    The base dir is /usr/local/env/graphics/meshlab/meshlab/src/distrib
    Current Plugins Dir is: /usr/local/env/graphics/meshlab/meshlab/src/distrib/plugins 
    Reading symbols for shared libraries .. done
    terminate called after throwing an instance of 'MeshLabXMLParsingException'
      what():  Error While parsing the XML filter plugin descriptors: We are trying to load a xml file that does not correspond to any dll or javascript code; please delete all the spurious xml files

    Program received signal SIGABRT, Aborted.
    0x957659f0 in __kill ()

    (gdb) bt 
    #0  0x957659f0 in __kill ()
    #1  0x95800bfc in abort ()
    #2  0x91c4de28 in __gnu_cxx::__verbose_terminate_handler ()
    #3  0x91c4b944 in __gxx_personality_v0 ()
    #4  0x91c4b9a8 in std::terminate ()
    #5  0x91c4bbe8 in __cxa_throw ()
    #6  0x00244fa8 in PluginManager::loadXMLPlugin ()
    #7  0x00247094 in PluginManager::loadPlugins ()
    #8  0x000161bc in MainWindow::MainWindow ()
    #9  0x00006b9c in main ()
    (gdb) 


Adding some debug, for XML loading find the file that causes the choke, its valid xml::

     xmllint  --pretty 1 /usr/local/env/graphics/meshlab/meshlab/src/distrib/plugins/filter_measure.xml 

Problem is an xml plugin file without corresponding dylib.  
Uncomment the subdir for that in meshlab_mini.pro, qmake, clang-kludge, make. 
Same story for filter_mutualinfoxml.

Now can bring up the GUI, but no collada import. Add that plugin.
Did g4_00.dae collada import from a gdb run. 
Observe that every face imported is being logged.  
Thats going to slowdown import substantially!

30 min to load::

    LOG: 0 Opened mesh /usr/local/env/geant4/geometry/gdml/gdml_dae_wrl/g4_00.dae in 1827120 msec
    LOG: 0 All files opened in 1835861 msec

Snapshot directory "." goes into the same dir as the mesh::

    simon:io_collada blyth$ cd  /usr/local/env/geant4/geometry/gdml/gdml_dae_wrl/
    simon:gdml_dae_wrl blyth$ open  snapshot00.png 

TODO

#. find out about qt logging and how to switch it off : for faster collada loading


X3D PLUGIN
-------------

Compiled it but no show in dialog ? Added debug to common/pluginmanager.cpp::

    checking: /usr/local/env/graphics/meshlab/meshlab/src/distrib/plugins/libio_x3d.dylib 
    Attempt pluginLoad: /usr/local/env/graphics/meshlab/meshlab/src/distrib/plugins/libio_x3d.dylib 
    pluginLoad failed: /usr/local/env/graphics/meshlab/meshlab/src/distrib/plugins/libio_x3d.dylib 

A recompilation fixes the plugin load::

    checking: /usr/local/env/graphics/meshlab/meshlab/src/distrib/plugins/libio_x3d.dylib 
    Attempt pluginLoad: /usr/local/env/graphics/meshlab/meshlab/src/distrib/plugins/libio_x3d.dylib 
    io pluginLoad: /usr/local/env/graphics/meshlab/meshlab/src/distrib/plugins/libio_x3d.dylib 

From the source, VRML gets translated into X3D first.

::

    simon:meshlab blyth$ find . -name '*.cpp' -exec grep -H VRML {} \;
    ./meshlab/src/meshlabplugins/io_base/baseio.cpp:        formatList << Format("VRML File Format"                                                 , tr("WRL"));
    ./meshlab/src/meshlabplugins/io_x3d/io_x3d.cpp: formatList << Format("X3D File Format - VRML encoding", tr("X3DV"));
    ./meshlab/src/meshlabplugins/io_x3d/io_x3d.cpp: formatList << Format("VRML 2.0 File Format", tr("WRL"));
    ./meshlab/src/meshlabplugins/io_x3d/vrml/Parser.cpp:                    case 9: s = coco_string_create(L"\"VRML\" expected"); break;
    ./meshlab/src/meshlabplugins/io_x3d/vrml/Scanner.cpp:   keywords.set(L"VRML", 9);

OSX GUI APP ISSUE
------------------

When launched in a GUI manner or with open the plugins are not found, so no DAE or WRL loading.
But the plugins are found when started in commandline way, and you get easy visibility to console::

   simon:MacOS blyth$ ./meshlab 


MESHLAB WINDOW TITLE  MeshLab v1.3.2_64bit
---------------------------------------------

Why the misnomer, are there large speedup factors to be had ? 


FUNCTIONS
-----------



EOU
}
meshlab-dir(){ echo $(local-base)/env/graphics/meshlab/meshlab/src ; }
meshlab-cd(){  cd $(meshlab-dir)/$1 ; }
meshlab-mate(){ mate $(meshlab-dir) ; }
meshlab-get(){
   local dir=$(dirname $(dirname $(meshlab-dir))) &&  mkdir -p $dir && cd $dir

   local tar=MeshLabSrc_AllInc_v132.tar
   echo  SF DOWNLOADING IS BROKEN : HAVE TO DO MANUALLY : mv ~/Downloads/$tar . 
}

meshlab-env(){      elocal- ; qt4- ; }
meshlab-launch(){
   meshlab-cd distrib/meshlab.app/Contents/MacOS
   ./meshlab
}

meshlab-find(){ find $(meshlab-dir) -name '*.cpp' -exec grep -H $1 {} \; }
meshlab-plugins-dir(){  echo $(meshlab-dir)/distrib/plugins ; }
meshlab-plugins-ls(){   ls -l $(meshlab-plugins-dir) ; }
meshlab-plugins-cd(){   cd $(meshlab-plugins-dir) ; }

meshlab-config(){ 
   #echo  meshlab_full.pro 
   echo  meshlab_mini.pro 
}

meshlab-external(){
   type $FUNCNAME
   meshlab-cd external
   qmake -recursive external.pro
}
meshlab-qmake(){
   type $FUNCNAME
   meshlab-cd 
   qmake -recursive $(meshlab-config)
   qt4-kludge
}
meshlab-make(){
   type $FUNCNAME
   meshlab-cd 
   make
}



