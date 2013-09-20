Rebuild Geant4 VRML2FILE Visualisation lib
=============================================

.. contents:: :local:

Geant4 Patch to see which solids give trouble
------------------------------------------------

Create patch::

    [blyth@belle7 LCG]$ cp -rf geant4.9.2.p01 geant4.9.2.p01.new
    [blyth@belle7 LCG]$ diff -rupN geant4.9.2.p01 geant4.9.2.p01.new 
    diff -rupN geant4.9.2.p01/source/visualization/VRML/src/G4VRML2SceneHandlerFunc.icc geant4.9.2.p01.new/source/visualization/VRML/src/G4VRML2SceneHandlerFunc.icc
    --- geant4.9.2.p01/source/visualization/VRML/src/G4VRML2SceneHandlerFunc.icc    2013-09-18 18:37:05.000000000 +0800
    +++ geant4.9.2.p01.new/source/visualization/VRML/src/G4VRML2SceneHandlerFunc.icc        2013-09-18 18:32:54.000000000 +0800
    @@ -186,6 +186,7 @@ void G4VRML2SCENEHANDLER::AddPrimitive(c
     
            // VRML codes are generated below
     
    +       G4cerr << "SCB ***** AddPrimitive(G4Polyhedron)" << pv_name << "\n";
            fDest << "#---------- SOLID: " << pv_name << "\n";
     
            if ( IsPVPickable() ) {
    [blyth@belle7 LCG]$ 
    [blyth@belle7 LCG]$ diff -rupN geant4.9.2.p01 geant4.9.2.p01.new > /data1/env/local/dyb/NuWa-trunk/lcgcmt/LCG_Builders/geant4/patches/geant4.9.2.p01.patch3 

Add the patch to LCG builders config script::

    [blyth@belle7 LCG]$ cat /data1/env/local/dyb/NuWa-trunk/lcgcmt/LCG_Builders/geant4/scripts/geant4_config.sh
    #!/bin/sh

    . ${LCG_BUILDPOLICYROOT_DIR}/scripts/common.sh
    untar

    goto ${LCG_srcdir}
    apply_patch geant4.9.2.p01.patch 1
    apply_patch geant4.9.2.p01.patch2 1
    apply_patch geant4.9.2.p01.patch3 1

Dybinst external geant4 build doing nothing::

    [blyth@belle7 dyb]$ ./dybinst trunk external geant4 

From LCG builders, see than need to move prior lib aside to force Geant4 to build::

    [blyth@belle7 cmt]$ pwd 
    /data1/env/local/dyb/NuWa-trunk/lcgcmt/LCG_Builders/geant4/cmt
    [blyth@belle7 cmt]$ vi ../../README.org 
    [blyth@belle7 cmt]$ cmt pkg_make
    Execute action pkg_make => sh -x /data1/env/local/dyb/NuWa-trunk/lcgcmt/LCG_Builders/geant4/scripts/geant4_make.sh
    + . /data1/env/local/dyb/NuWa-trunk/lcgcmt/LCG_Builders/LCG_BuildPolicy/scripts/common.sh
    + CPPVERBOSE=1
    + export CPPVERBOSE
    + goto /data1/env/local/dyb/NuWa-trunk/../external/build/LCG/geant4.9.2.p01/source
    + dir=/data1/env/local/dyb/NuWa-trunk/../external/build/LCG/geant4.9.2.p01/source
    + '[' -n /data1/env/local/dyb/NuWa-trunk/../external/build/LCG/geant4.9.2.p01/source ']'
    + shift
    + cmd cd /data1/env/local/dyb/NuWa-trunk/../external/build/LCG/geant4.9.2.p01/source
    + info 'running command: cd /data1/env/local/dyb/NuWa-trunk/../external/build/LCG/geant4.9.2.p01/source'
    + '[' -n 'running command: cd /data1/env/local/dyb/NuWa-trunk/../external/build/LCG/geant4.9.2.p01/source' ']'
    + msg='running command: cd /data1/env/local/dyb/NuWa-trunk/../external/build/LCG/geant4.9.2.p01/source'
    + shift
    + echo 'geant4: "running command: cd /data1/env/local/dyb/NuWa-trunk/../external/build/LCG/geant4.9.2.p01/source"'
    geant4: "running command: cd /data1/env/local/dyb/NuWa-trunk/../external/build/LCG/geant4.9.2.p01/source"
    + cd /data1/env/local/dyb/NuWa-trunk/../external/build/LCG/geant4.9.2.p01/source
    + check 'running command: cd /data1/env/local/dyb/NuWa-trunk/../external/build/LCG/geant4.9.2.p01/source'
    + err=0
    + msg='running command: cd /data1/env/local/dyb/NuWa-trunk/../external/build/LCG/geant4.9.2.p01/source'
    + '[' -n 'running command: cd /data1/env/local/dyb/NuWa-trunk/../external/build/LCG/geant4.9.2.p01/source' ']'
    + shift
    + '[' 0 '!=' 0 ']'
    + '[' '!' -f /data1/env/local/dyb/NuWa-trunk/../external/build/LCG/geant4.9.2.p01/lib/Linux-g++/libG4run.so ']'
    + '[' '!' -f /data1/env/local/dyb/NuWa-trunk/../external/build/LCG/geant4.9.2.p01/lib/Linux-g++/libname.map ']'
    + '[' '!' -f /data1/env/local/dyb/NuWa-trunk/../external/build/LCG/geant4.9.2.p01/include/G4Version.hh ']'
    [blyth@belle7 cmt]$ 
    [blyth@belle7 cmt]$ 
    [blyth@belle7 cmt]$ l /data1/env/local/dyb/NuWa-trunk/../external/build/LCG/geant4.9.2.p01/lib/Linux-g++/libG4run.so
    -rwxrwxr-x 1 blyth blyth 3558478 Feb 16  2011 /data1/env/local/dyb/NuWa-trunk/../external/build/LCG/geant4.9.2.p01/lib/Linux-g++/libG4run.so
    [blyth@belle7 cmt]$ 


Then install::

    [blyth@belle7 dyb]$ cd NuWa-trunk/lcgcmt/LCG_Builders/geant4/cmt && cmt config && . setup.sh 
    [blyth@belle7 cmt]$ cmt pkg_install 
    Execute action pkg_install => sh -x /data1/env/local/dyb/NuWa-trunk/lcgcmt/LCG_Builders/geant4/scripts/geant4_install.sh
    + echo 'geant4: installing code'
    geant4: installing code
    + cd /data1/env/local/dyb/NuWa-trunk/../external/build/LCG/geant4.9.2.p01
    + mkdir -p /data1/env/local/dyb/NuWa-trunk/../external/geant4/4.9.2.p01/i686-slc5-gcc41-dbg
    + for dir in lib include
    + target=/data1/env/local/dyb/NuWa-trunk/../external/geant4/4.9.2.p01/i686-slc5-gcc41-dbg/lib
    + '[' -d /data1/env/local/dyb/NuWa-trunk/../external/geant4/4.9.2.p01/i686-slc5-gcc41-dbg/lib ']'
    + echo 'geant4: /data1/env/local/dyb/NuWa-trunk/../external/geant4/4.9.2.p01/i686-slc5-gcc41-dbg/lib already exists, remove to force reinstall'
    geant4: /data1/env/local/dyb/NuWa-trunk/../external/geant4/4.9.2.p01/i686-slc5-gcc41-dbg/lib already exists, remove to force reinstall
    + for dir in lib include


    [blyth@belle7 cmt]$ ll /data1/env/local/dyb/NuWa-trunk/../external/geant4/4.9.2.p01/i686-slc5-gcc41-dbg/lib/
    total 390376
    -rwxrwxr-x 1 blyth blyth  1803700 Feb 16  2011 libG4globman.so
    -rwxrwxr-x 1 blyth blyth   642186 Feb 16  2011 libG4hepnumerics.so
    -rwxrwxr-x 1 blyth blyth  2327915 Feb 16  2011 libG4intercoms.so
    ..

    [blyth@belle7 cmt]$ mv /data1/env/local/dyb/NuWa-trunk/../external/geant4/4.9.2.p01/i686-slc5-gcc41-dbg/lib /data1/env/local/dyb/NuWa-trunk/../external/geant4/4.9.2.p01/i686-slc5-gcc41-dbg/lib.prior
    [blyth@belle7 cmt]$ cmt pkg_install


Argh, thats building all libs. Hope thats fixed in cmake geant4. Suspect could just build the relevant one::

    cd /data1/env/local/dyb/external/build/LCG/geant4.9.2.p01/source/visualization/VRML
    make 

Nope need to pass in some env::

    [blyth@belle7 VRML]$ l ../../../lib/Linux-g++/libG4VRML* 
    -rw-rw-r-- 1 blyth blyth 2502820 Sep 18 20:16 ../../../lib/Linux-g++/libG4VRML.a
    -rwxrwxr-x 1 blyth blyth 2080324 Sep 18 19:33 ../../../lib/Linux-g++/libG4VRML.so
    [blyth@belle7 VRML]$ rm ../../../lib/Linux-g++/libG4VRML.a
    [blyth@belle7 VRML]$ 
    [blyth@belle7 VRML]$ make CLHEP_BASE_DIR=/data1/env/local/dyb/external/clhep/2.0.4.2/i686-slc5-gcc41-dbg G4SYSTEM=Linux-g++

::

    [blyth@belle7 VRML]$ make CLHEP_BASE_DIR=/data1/env/local/dyb/external/clhep/2.0.4.2/i686-slc5-gcc41-dbg G4SYSTEM=Linux-g++ G4LIB_BUILD_SHARED=1
    make: Nothing to be done for `lib'.
    [blyth@belle7 VRML]$ rm ../../../lib/Linux-g++/libG4VRML.so && make CLHEP_BASE_DIR=/data1/env/local/dyb/external/clhep/2.0.4.2/i686-slc5-gcc41-dbg/include G4SYSTEM=Linux-g++ G4LIB_BUILD_SHARED=1
    Creating shared library ../../../lib/Linux-g++/libG4VRML.so ...


The install is all libs or nothing, so manual copy::

    [blyth@belle7 VRML]$ cp ../../../lib/Linux-g++/libG4VRML.so /data1/env/local/dyb/NuWa-trunk/../external/geant4/4.9.2.p01/i686-slc5-gcc41-dbg/lib/libG4VRML.so


Annotated Traverse
---------------------

Now can see which volumes cause the warnings::

    [blyth@belle7 export]$ nuwa.py -G $XMLDETDESCROOT/DDDB/dayabay.xml -n1 -m export > exportdbg.txt


Output buffering mismatch ? Leaves me uncertain that the error goes with the preceeding volume::

    SCB ***** AddPrimitive(G4Polyhedron)/dd/Geometry/CalibrationSources/lvWallLedSourceAssy#pvWallLedDiffuserBall.1000
    SCB ***** AddPrimitive(G4Polyhedron)/dd/Geometry/CalibrationSources/lvWallLedSourceAssy#pvWallLedAcryBooleanProcessor: boolean operation failed
    BooleanProcessor: boolean operation failed
    BooleanProcessor: boolean operation failed
    ...
    BooleanProcessor: boolean operation failed
    BooleanProcessor: boolean operation failed
    licRod.1001
    SCB ***** AddPrimitive(G4Polyhedron)/dd/Geometry/AD/lvSST#lvCenterCalibHoleSST.1001
    SCB ***** AddPrimitive(G4Polyhedron)/dd/Geometry/AD/lvSST#pvOffCenterCalibHoleSST.1002

Using std::cerr for the volnames to match most of the warnings does better::

    [blyth@belle7 VRML]$ touch src/G4VRML2FileSceneHandler.cc
    [blyth@belle7 VRML]$ rm ../../../lib/Linux-g++/libG4VRML.so 
    [blyth@belle7 VRML]$ make CLHEP_BASE_DIR=/data1/env/local/dyb/external/clhep/2.0.4.2/i686-slc5-gcc41-dbg G4SYSTEM=Linux-g++ G4LIB_BUILD_SHARED=1   
    Making dependency for file src/G4VRML2FileSceneHandler.cc ...
    Compiling G4VRML2FileSceneHandler.cc ...
    Creating shared library ../../../lib/Linux-g++/libG4VRML.so ...

::

    [blyth@belle7 export]$ nuwa.py -G $XMLDETDESCROOT/DDDB/dayabay.xml -n1 -m export > exportdbg.txt 2>&1


