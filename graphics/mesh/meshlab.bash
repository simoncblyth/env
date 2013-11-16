# === func-gen- : graphics/mesh/meshlab fgp graphics/mesh/meshlab.bash fgn meshlab fgh graphics/mesh
meshlab-src(){      echo graphics/mesh/meshlab.bash ; }
meshlab-source(){   echo ${BASH_SOURCE:-$(env-home)/$(meshlab-src)} ; }
meshlab-vi(){       vi $(meshlab-source) ; }
meshlab-env(){      elocal- ; }
meshlab-usage(){ cat << EOU
MESHLAB
========

Qt based GUI for mesh viewing/manipulation

* http://meshlab.sourceforge.net/
* http://sourceforge.net/apps/mediawiki/meshlab/index.php?title=Interacting_with_the_mesh

Meshpad
-------

* http://meshlabstuff.blogspot.tw/ 
* https://itunes.apple.com/us/app/meshlab-for-ios/id451944013
* http://www.meshpad.org/

iOS source ? (hmm GPL)
~~~~~~~~~~~~~~~~~~~~~~~

* http://sourceforge.net/p/meshlab/discussion/499533/thread/836b4da1/

The source code of MeshLab for iOS and MeshLab for Android is not available for
the general public. We can provide customized version of it, or license the
viewing component under a commercial agreement.

Forum
------

* http://sourceforge.net/p/meshlab/discussion/499533

Observations
-------------

#. supports VRML/X3D/STL import and export 
#. documentation is sparse

Intro
-------

* http://sourceforge.net/apps/mediawiki/meshlab/index.php?title=Main_Page

MeshLab is a advanced mesh processing system, for the automatic and user
assisted editing, cleaning, filtering converting and rendering of large
unstructured 3D triangular meshes. MeshLab is actively developed by the a small
group of people at the Visual Computing Lab at the ISTI - CNR institute, a
large group of university students and some great developers from the rest of
the world. For the basic mesh processing tasks and for the internal data
structures the system relies on the GPL VCG library.

Question : consequences of *unstructured* ? 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* need to identify surfaces/materials, which correspond to many tris


VCG
----

* http://vcg.isti.cnr.it/~cignoni/newvcglib/html/

The Visualization and Computer Graphics Library (VCG for short) is a open
source portable C++ templated library for manipulation, processing and
displaying with OpenGL of triangle and tetrahedral meshes.

The library, composed by more than 100k lines of code, is released under the
GPL license, and it is the base of most of the software tools of the Visual
Computing Lab of the Italian National Research Council Institute ISTI
(http://vcg.isti.cnr.it), like metro and MeshLab.


Pre-requisites
---------------

* http://meshlab.sourceforge.net/wiki/index.php/Compiling

* Qt 4.8 (note that Qt 4.7 are required, Qt versions < 4.7 could not compile).
  Current version of MeshLab compiles well against Qt 4.7.4.


Repo
----

* http://svn.code.sf.net/p/meshlab/code/trunk/meshlab/src/

::

    svn checkout svn://svn.code.sf.net/p/meshlab/code/trunk meshlab-code
    svn checkout http://svn.code.sf.net/p/meshlab/code/trunk meshlab-code


Download
----------

::

   mv ~/Downloads/MeshLabSrc_AllInc_v132.tar .   ## WARNING : exploding tarball
  
::

    simon:meshlab blyth$ l
    total 94552
    drwxr-xr-x@ 7 blyth  wheel       238 22 Aug 13:16 vcglib
    drwxr-xr-x@ 3 blyth  wheel       102 22 Aug 13:15 meshlab
    -rw-r--r--@ 1 blyth  staff  48404480 22 Aug 13:07 MeshLabSrc_AllInc_v132.tar
    -rw-r--r--@ 1 blyth  wheel       150  3 Aug  2012 how_to_compile.txt
    simon:meshlab blyth$ pwd
    /usr/local/env/graphics/mesh/graphics/meshlab


VRML/X3D
--------------

From the source, VRML gets translated into X3D first.

::

    simon:meshlab blyth$ find . -name '*.cpp' -exec grep -H VRML {} \;
    ./meshlab/src/meshlabplugins/io_base/baseio.cpp:        formatList << Format("VRML File Format"                                                 , tr("WRL"));
    ./meshlab/src/meshlabplugins/io_x3d/io_x3d.cpp: formatList << Format("X3D File Format - VRML encoding", tr("X3DV"));
    ./meshlab/src/meshlabplugins/io_x3d/io_x3d.cpp: formatList << Format("VRML 2.0 File Format", tr("WRL"));
    ./meshlab/src/meshlabplugins/io_x3d/vrml/Parser.cpp:                    case 9: s = coco_string_create(L"\"VRML\" expected"); break;
    ./meshlab/src/meshlabplugins/io_x3d/vrml/Scanner.cpp:   keywords.set(L"VRML", 9);



Navigation with Intel Mac Laptop
---------------------------------

shift-cmd-H
             return to home position 
double click
             change center of rotation
one finger drag
             move viewpoint about rotation position 

two-finger 
             dolly forward/backward
shift+two-finger
             change camera field of view
cmd+two-finger
             change near clipping place

shift+cmd+one-finger
             change light direction         

opt/alt-return
             toggle fullscreen mode



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

     

EOU
}
meshlab-dir(){ echo $(local-base)/env/graphics/meshlab/meshlab ; }
meshlab-cd(){  cd $(meshlab-dir); }
meshlab-mate(){ mate $(meshlab-dir) ; }
meshlab-get(){
   local dir=$(dirname $(meshlab-dir)) &&  mkdir -p $dir && cd $dir

   local tar=MeshLabSrc_AllInc_v132.tar
   echo  SF DOWNLOADING IS BROKEN : HAVE TO DO MANUALLY : mv ~/Dowloads/$tar . 

}
