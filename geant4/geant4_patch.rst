Geant4 Patch
============

LCG Geant4 patch machinery within NuWa
----------------------------------------

* NuWa-trunk/lcgcmt/LCG_Builders/geant4/scripts/geant4_config.sh

::

    [blyth@belle7 scripts]$ cat geant4_config.sh 
    #!/bin/sh

    . ${LCG_BUILDPOLICYROOT_DIR}/scripts/common.sh
    untar

    goto ${LCG_srcdir}
    apply_patch geant4.9.2.p01.patch 1
    apply_patch geant4.9.2.p01.patch2 1
    apply_patch geant4.9.2.p01.patch3 1


Material Property Introspection patch
---------------------------------------

::

    [blyth@belle7 include]$ pwd
    /data1/env/local/dyb/external/build/LCG/geant4.9.2.p01/source/materials/include

    [blyth@belle7 include]$ diff G4MaterialPropertiesTable.hh.orig G4MaterialPropertiesTable.hh
    136a137,149
    > 
    >   // copied from Geant4 future
    >   public:  // without description
    > 
    >     const std::map< G4String, G4MaterialPropertyVector*, std::less<G4String> >*
    >     GetPropertiesMap() const { return &MPT; }
    >     const std::map< G4String, G4double, std::less<G4String> >*
    >     GetPropertiesCMap() const { return &MPTC; }
    >     // Accessors required for persistency purposes
    > 


Geant4 Rebuild
----------------

Simple rebuild is too quick, doing nothing::

    [blyth@belle7 dyb]$ ./dybinst trunk external geant4


    Tue Feb 18 10:43:06 CST 2014
    Start Logging to /data1/env/local/dyb/dybinst-20140218-104306.log (or dybinst-recent.log)


    Starting dybinst commands: external

    Stage: "external"... 

    Found CMTCONFIG="i686-slc5-gcc41-dbg" from lcgcmt
    Checking your CMTCONFIG="i686-slc5-gcc41-dbg"...
    ...ok.

    dybinst-external: installing packages: geant4

    Installing external packages, this will take a while.  Go get coffee...
      Installing geant4 ... done with geant4
    [blyth@belle7 dyb]$ 



From the log::

    [blyth@belle7 dyb]$ grep ^geant4: /data1/env/local/dyb/dybinst-20140218-104306.log
    geant4: running "cmt pkg_get"
    geant4: running "cmt pkg_config"
    geant4: "using file from LCG_tarfilename="geant4.9.2.p01.tar.gz""
    geant4: "running command: cd /data1/env/local/dyb/NuWa-trunk/../external/build/LCG"
    geant4: "source directory exists, to re-untar remove "/data1/env/local/dyb/NuWa-trunk/../external/build/LCG/geant4.9.2.p01""
    geant4: "running command: cd /data1/env/local/dyb/NuWa-trunk/lcgcmt/LCG_Builders/geant4/cmt"
    geant4: "running command: cd /data1/env/local/dyb/NuWa-trunk/../external/build/LCG/geant4.9.2.p01"
    geant4: "Already applied patch "/data1/env/local/dyb/NuWa-trunk/lcgcmt/LCG_Builders/geant4/patches/geant4.9.2.p01.patch""
    geant4: "Already applied patch "/data1/env/local/dyb/NuWa-trunk/lcgcmt/LCG_Builders/geant4/patches/geant4.9.2.p01.patch2""
    geant4: "Already applied patch "/data1/env/local/dyb/NuWa-trunk/lcgcmt/LCG_Builders/geant4/patches/geant4.9.2.p01.patch3""
    geant4: running "cmt pkg_make"
    geant4: "running command: cd /data1/env/local/dyb/NuWa-trunk/../external/build/LCG/geant4.9.2.p01/source"
    geant4: running "cmt pkg_install"
    geant4: installing code
    geant4: /data1/env/local/dyb/NuWa-trunk/../external/geant4/4.9.2.p01/i686-slc5-gcc41-dbg/lib already exists, remove to force reinstall
    geant4: /data1/env/local/dyb/NuWa-trunk/../external/geant4/4.9.2.p01/i686-slc5-gcc41-dbg/include already exists, remove to force reinstall
    geant4: installing data
    [blyth@belle7 dyb]$ 

The make step::

    [blyth@belle7 dyb]$ cat /data1/env/local/dyb/NuWa-trunk/lcgcmt/LCG_Builders/geant4/scripts/geant4_make.sh
    #!/bin/sh

    . ${LCG_BUILDPOLICYROOT_DIR}/scripts/common.sh

    CPPVERBOSE=1
    export CPPVERBOSE

    # Geant4's make is a bit more than just "make" so spell it out

    goto $LCG_srcdir/source
    if [ ! -f ${G4INSTALL}/lib/$G4SYSTEM/libG4run.so ] ; then
        cmd make 
    fi
    if [ ! -f ${G4INSTALL}/lib/$G4SYSTEM/libname.map ] ; then
        cmd make libmap
    fi
    if [ ! -f ${G4INSTALL}/include/G4Version.hh ] ; then
        cmd make includes
    fi

             
Jump in and build::

    fenv  # pick up basis env
    cd /data1/env/local/dyb/NuWa-trunk/lcgcmt/LCG_Builders/geant4/cmt
    cmt config
    . setup.sh

Detects libG4run.so and does nothing::

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
    [blyth@belle7 cmt]$  l /data1/env/local/dyb/NuWa-trunk/../external/build/LCG/geant4.9.2.p01/lib/Linux-g++/libG4run.so
    -rwxrwxr-x 1 blyth blyth 3558478 Sep 18 19:27 /data1/env/local/dyb/NuWa-trunk/../external/build/LCG/geant4.9.2.p01/lib/Linux-g++/libG4run.so
    [blyth@belle7 cmt]$     


Removing the libG4run.so coaxes the build into action, a full build it seems::

    [blyth@belle7 cmt]$ mv /data1/env/local/dyb/NuWa-trunk/../external/build/LCG/geant4.9.2.p01/lib/Linux-g++/libG4run.so /data1/env/local/dyb/NuWa-trunk/../external/build/LCG/geant4.9.2.p01/lib/Linux-g++/libG4run.so.rebuild
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
    + cmd make
    + info 'running command: make'
    + '[' -n 'running command: make' ']'
    + msg='running command: make'
    + shift
    + echo 'geant4: "running command: make"'
    geant4: "running command: make"
    + make
    *************************************************************
     Installation Geant4 version : geant4-09-02-patch-01 
     Copyright (C) 1994-2009 Geant4 Collaboration                            
    *************************************************************
    Creating shared library /data1/env/local/dyb/NuWa-trunk/../external/build/LCG/geant4.9.2.p01/lib/Linux-g++/libG4globman.so ...
    Creating shared library /data1/env/local/dyb/NuWa-trunk/../external/build/LCG/geant4.9.2.p01/lib/Linux-g++/libG4hepnumerics.so ...
    make[1]: Nothing to be done for `lib'.
    make[1]: Nothing to be done for `lib'.
    Making dependency for file src/G4SandiaTable.cc ...
    Making dependency for file src/G4NistMessenger.cc ...
    Making dependency for file src/G4NistMaterialBuilder.cc ...
    Making dependency for file src/G4NistManager.cc ...
    Making dependency for file src/G4MaterialPropertiesTable.cc ...
    ...


Record the rebuild method in::

   g4-libs-rebuild
   g4-includes-rebuild



