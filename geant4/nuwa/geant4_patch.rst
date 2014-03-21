Geant4 Patch
============

Need for patching
------------------

To get newer GDML with material and surface properties to work 
with old geant4 requires

* expose G4MaterialPropertiesTable maps

* API adjustments to bring new geant4 API for G4MaterialPropertyVector (which typedefs G4PhysicsOrderedFreeVector)
  to older geant4 G4MaterialPropertyVector

Notes
~~~~~~~

Compared new geant4 GDML with the version used with NuWa 

* /usr/local/env/geant4/geant4.10.00.b01/source/persistency/gdml/include

::

    src/G4DAEWriteMaterials.cc:144: error: cannot convert 
    'const std::map<G4String, G4MaterialPropertyVector*, std::less<G4String>, std::allocator<std::pair<const G4String, G4MaterialPropertyVector*> > >*' to 
    'const std::map<G4String, G4PhysicsOrderedFreeVector*, std::less<G4String>, std::allocator<std::pair<const G4String, G4PhysicsOrderedFreeVector*> > >*' in initialization

Future geant4 gets rid of the deficient G4MaterialPropertyVector typedefing from G4PhysicsOrderedFreeVector.
/usr/local/env/geant4/geant4.10.00.b01/source/materials/include/G4MaterialPropertyVector.hh::

     56 #include "G4PhysicsOrderedFreeVector.hh"
     ..
     62 typedef G4PhysicsOrderedFreeVector G4MaterialPropertyVector;




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


`LCG_srcdir`::

    [blyth@belle7 geant4.9.2.p01]$ ll 
    total 324
    -rw-r--r--  1 blyth blyth   4029 Mar 16  2009 LICENSE
    -rwxr-xr-x  1 blyth blyth 142639 Mar 16  2009 Configure
    drwxr-xr-x  2 blyth blyth   4096 Mar 16  2009 ReleaseNotes
    drwxr-xr-x  5 blyth blyth   4096 Mar 16  2009 examples
    drwxr-xr-x  4 blyth blyth   4096 Mar 16  2009 environments
    drwxr-xr-x  4 blyth blyth   4096 Mar 16  2009 config
    -rw-rw-r--  1 blyth blyth      0 Mar  3 16:43 .geant4.9.2.p01.patch2
    -rw-rw-r--  1 blyth blyth      0 Mar  3 16:43 .geant4.9.2.p01.patch
    drwxrwxr-x  3 blyth blyth   4096 Mar  3 16:43 lib
    drwxrwxr-x  2 blyth blyth   4096 Mar  3 16:43 bin
    drwxr-xr-x 11 blyth blyth   4096 Mar  3 16:43 .
    drwxrwxr-x  3 blyth blyth   4096 Mar  3 16:43 tmp
    drwxrwxr-x  2 blyth blyth 135168 Mar  3 17:15 include
    drwxr-xr-x 22 blyth blyth   4096 Mar  4 17:39 source
    drwxrwxr-x 30 blyth blyth   4096 Mar  4 18:03 ..
    [blyth@belle7 geant4.9.2.p01]$ pwd
    /data1/env/local/dyb/external/build/LCG/geant4.9.2.p01

    [blyth@belle7 geant4.9.2.p01]$ find . -name '*.orig' -exec ls -l {} \;
    -rw-r--r-- 1 blyth blyth 10395 Mar 16  2009 ./include/G4OpBoundaryProcess.hh.orig
    -rw-r--r-- 1 blyth blyth 10395 Mar 16  2009 ./source/processes/optical/include/G4OpBoundaryProcess.hh.orig
    [blyth@belle7 geant4.9.2.p01]$ 


Suspect that `G4OpBoundaryProcess.hh.orig` suggests suspect patch application.



LCG apply_patch
-----------------

* /data1/env/local/dyb/NuWa-trunk/lcgcmt/LCG_Builders/LCG_BuildPolicy/scripts/common.sh 

::

    108 # Apply given patch with optional strip level (default is 0).  If the
    109 # patchfile is not found it is looked for as a path relative to
    110 # Package/patches/ assuming the running script is in Package/scripts/.
    111 apply_patch () {
    112     file=$1 ;
    113     if [ -n "$1" ] ; then shift; fi
    114     level=${1-"0"}
    115     if [ -n "$1" ] ; then shift; fi
    116 
    117     if [ ! -f "$file" ] ; then
    118     file=$(dirname $(dirname $0))/patches/$file
    119     fi
    120     if [ ! -f "$file" ] ; then
    121     fatal "failed to find patch file $file"
    122     fi
    123 
    124     been_here=".$(basename $file)"
    125     if [ -f "$been_here" ] ; then
    126     info "Already applied patch \"$file\""
    127     return
    128     fi
    129 
    130     # need quotes since using redirection
    131     cmd "patch -N -p$level -i $file"
    132     touch $been_here
    133 }


man patch
~~~~~~~~~~~

* `-R` looks dangerous


`-N` or `--forward`
        Ignore patches that seem to be reversed or already applied.  See also -R.

`-R` or `--reverse`
        Assume that this patch was created with the old and new files
        swapped.  (Yes, I'm afraid that does happen occasionally, human nature being
        what it is.)  patch attempts to swap each hunk around before applying  it.
        Rejects  come out in the swapped format.  The -R option does not work with ed
        diff scripts because there is too little information to reconstruct the reverse
        operation.

        If  the first hunk of a patch fails, patch reverses the hunk to see
        if it can be applied that way.  If it can, you are asked if you want to have
        the -R option set.  If it can't, the patch continues to be applied normally.
        (Note: this method cannot detect a reversed patch if it is a normal diff and if
        the first command is an append (i.e. it should have been a delete) since
        appends always succeed, due to the fact  that  a  null  context  matches
        anywhere.  Luckily, most patches add or change lines rather than delete them,
        so most reversed normal diffs begin with a delete, which fails, triggering the
        heuristic.)





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


Usage in future geant4.10 GDML persisting
--------------------------------------------

::

    g4pb:src blyth$ grep GetPropertiesMap *.cc
    G4GDMLWriteMaterials.cc:                 std::less<G4String> >* pmap = ptable->GetPropertiesMap();
    g4pb:src blyth$ pwd
    /usr/local/env/geant4/geant4.10.00.b01/source/persistency/gdml/src


Backport GDML Property writing 
--------------------------------

Not so simple due to geant4 change from use of G4MaterialPropertyVector to G4PhysicsOrderedFreeVector, 
so as stuck with older geant4 need to workaround limitations of the old G4MaterialPropertyVector.

What planet are the authors of G4MaterialPropertyVector from, no API to access NumEntries ? 
Its simpler to patch than to kludge iterate to count entries, plus patch to match 
Geant4 future API.



2014/02/19 compare G4MaterialPropertyVector with G4PhysicsVector
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    [blyth@cms01 geant4.9.2.p01]$ cd source/
    [blyth@cms01 source]$ find . -name 'G4MaterialPropertyVector.cc'
    ./materials/src/G4MaterialPropertyVector.cc
    [blyth@cms01 source]$ find . -name 'G4MaterialPropertyVector.hh'
    ./materials/include/G4MaterialPropertyVector.hh

    [blyth@cms01 source]$ find . -name 'G4PhysicsFreeVector.cc'
    ./global/management/src/G4PhysicsFreeVector.cc
    [blyth@cms01 source]$ find . -name 'G4PhysicsFreeVector.hh'
    ./global/management/include/G4PhysicsFreeVector.hh

    [blyth@cms01 source]$ find . -name 'G4PhysicsOrderedFreeVector.hh'
    ./global/management/include/G4PhysicsOrderedFreeVector.hh
    [blyth@cms01 source]$ 
    [blyth@cms01 source]$ find . -name 'G4PhysicsOrderedFreeVector.cc'
    ./global/management/src/G4PhysicsOrderedFreeVector.cc
    [blyth@cms01 source]$ 


2014/02/19 Backport G4PhysicsVector API to patch G4MaterialPropertyVector
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All this to get MPV to spill the beans::

    [blyth@cms01 source]$ vi materials/src/G4MaterialPropertyVector.cc  materials/include/G4MaterialPropertyVector.hh


    public: // SCB getting MPV to spill the beans  

       size_t GetVectorLength() const ;
       G4double Energy(const size_t binNumber) const ;
       G4double operator[](const size_t binNumber) const;



    // SCB getting MPV to spill the beans

    size_t G4MaterialPropertyVector::GetVectorLength() const 
    {
        return NumEntries ;
    }

    G4double G4MaterialPropertyVector::Energy(const size_t binNumber) const 
    {
        G4MPVEntry* mpv = MPV[binNumber] ;
        return mpv->GetPhotonEnergy();
    }  

    G4double G4MaterialPropertyVector::operator[](const size_t binNumber) const
    {
        G4MPVEntry* mpv = MPV[binNumber] ;
        return mpv->GetProperty();
    }


::

    [blyth@belle7 source]$ cp materials/src/G4MaterialPropertyVector.cc  materials/src/G4MaterialPropertyVector.cc.orig
    [blyth@belle7 source]$ cp materials/include/G4MaterialPropertyVector.hh materials/include/G4MaterialPropertyVector.hh.orig
    [blyth@belle7 source]$ vi materials/src/G4MaterialPropertyVector.cc  materials/include/G4MaterialPropertyVector.hh
    [blyth@belle7 source]$ g4-
    [blyth@belle7 source]$ g4-libs-rebuild      ## tedious full rebuild of all g4 libs, cmake is in geant4 future so no motivation to improve this back here
    ...
    [blyth@belle7 source]$ g4-includes-rebuild  ## also needed, to use the added APIs
    ...


2014/02/19 Incomplete Rebuild
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Runtime fail::

    788 G4DAE: Writing library_materials...
    789 G4DAE: Writing structure/library_nodes...
    790 G4DAE: Writing library_visual_scenes...
    791 python: symbol lookup error: /data1/env/local/dyb/NuWa-trunk/../external/geant4/4.9.2.p01/i686-slc5-gcc41-dbg/lib/libG4DAE.so: undefined symbol: _ZNK24G4MaterialPropertyVector15GetVectorLengthEv

Hmm missing the install step, as libs in /data1/env/local/dyb/external/geant4/4.9.2.p01/i686-slc5-gcc41-dbg/lib are old

Add in `g4-install-rebuild` to do that after the below investigations of the dybinst geant4 build mechanism.

But subsequently another runtime fail from missing libG4gdml.so::

     48 DetectorDataSvc                    SUCCESS Detector description database: /data1/env/local/dyb/NuWa-trunk/dybgaudi/Detector/XmlDetDesc/DDDB/dayabay.xml
     49 EventClockSvc.FakeEventTime           INFO Event times generated from 0 with steps of 0
     50 Generator                             INFO Added gen tool GtTransformTool/onemuonTransformer
     51 AlgorithmManager                     ERROR Algorithm of type GiGaInputStream is unknown (No factory available).
     52 AlgorithmManager                     ERROR libG4gdml.so: cannot open shared object file: No such file or directory
     53 AlgorithmManager                     ERROR More information may be available by setting the global jobOpt "ReflexPluginDebugLevel" to 1
     54 GaudiSequencer                     WARNING Unable to find or create GiGaInputStream
     55 AlgorithmManager                     ERROR Algorithm of type DsPushKine is unknown (No factory available).
     56 AlgorithmManager                     ERROR libG4gdml.so: cannot open shared object file: No such file or directory
     57 AlgorithmManager                     ERROR More information may be available by setting the global jobOpt "ReflexPluginDebugLevel" to 1
     58 GaudiSequencer                     WARNING Unable to find or create DsPushKine
     59 AlgorithmManager                     ERROR Algorithm of type DsPullEvent is unknown (No factory available).
     60 AlgorithmManager                     ERROR libG4gdml.so: cannot open shared object file: No such file or directory
     61 AlgorithmManager                     ERROR More information may be available by setting the global jobOpt "ReflexPluginDebugLevel" to 1
     62 GaudiSequencer                     WARNING Unable to find or create DsPullEvent
     63 GaudiSequencer                        INFO Member list:

Compare libs::

    [blyth@belle7 4.9.2.p01]$ ( cd i686-slc5-gcc41-dbg/lib/ ; ls -1 *.so ) > new.so
    [blyth@belle7 4.9.2.p01]$ ( cd i686-slc5-gcc41-dbg.prior/lib/ ; ls -1 *.so ) > old.so
    [blyth@belle7 4.9.2.p01]$ diff old.so new.so
    6a7
    > libG4DAEFILE.so
    22d22
    < libG4gdml.so
    [blyth@belle7 4.9.2.p01]$ 

Get dirty::

    [blyth@belle7 4.9.2.p01]$ cp i686-slc5-gcc41-dbg.prior/lib/libG4gdml.so i686-slc5-gcc41-dbg/lib/
    [blyth@belle7 4.9.2.p01]$ 

This succeeds to write properties to the DAE, need some veracity checking::

     70373     <material id="__dd__Materials__Acrylic0xa7b6b48">
     70374       <instance_effect url="#__dd__Materials__Acrylic_fx_0xa7b6b48"/>
     70375       <extra>
     70376         <matrix coldim="2" name="ABSLENGTH0xa7b4d78" values="1.55e-06 8000 1.61e-06 8000 2.07e-06 8000 2.48e-06 8000 3.76e-06 8000 4.13e-06 8000 6.2e-06 0.008 1.033e-05 0.008 1.55e-05 0.008"/>
     70377         <property name="ABSLENGTH" ref="ABSLENGTH0xa7b4d78"/>
     70378         <matrix coldim="2" name="RAYLEIGH0xa7b4da8" values="1.55e-06 500000 1.7714e-06 300000 2.102e-06 170000 2.255e-06 100000 2.531e-06 62000 2.884e-06 42000 3.024e-06 30000 4.133e-06 7600 6.2e-06 850        1.033e-05 850 1.55e-05 850"/>
     70379         <property name="RAYLEIGH" ref="RAYLEIGH0xa7b4da8"/>
     70380         <matrix coldim="2" name="RINDEX0xa504f20" values="1.55e-06 1.4878 1.79505e-06 1.4895 2.10499e-06 1.4925 2.27077e-06 1.4946 2.55111e-06 1.4986 2.84498e-06 1.5022 3.06361e-06 1.5065 4.13281e-06 1.       5358 6.2e-06 1.6279 6.526e-06 1.627 6.889e-06 1.5359 7.294e-06 1.5635 7.75e-06 1.793 8.267e-06 1.7199 8.857e-06"/>
     70381         <property name="RINDEX" ref="RINDEX0xa504f20"/>
     70382       </extra>
     70383     </material>

Possible truncation::

     70497     <material id="__dd__Materials__ESR0xa56f4b0">
     70498       <instance_effect url="#__dd__Materials__ESR_fx_0xa56f4b0"/>
     70499       <extra>
     70500         <matrix coldim="2" name="ABSLENGTH0xa8080f8" values="1.55e-06 0.001 1.63e-06 0.001 1.68e-06 0.001 1.72e-06 0.001 1.77e-06 0.001 1.82e-06 0.001 1.88e-06 0.001 1.94e-06 0.001 2e-06 0.001 2.07e-06        0.001 2.14e-06 0.001 2.21e-06 0.001 2.3e-06 0.001 2.38e-06 0.001 2.48e-06 0.001 2.58e-06 0.001 2.7e-06 0.001 2.82e"/>
     70501         <property name="ABSLENGTH" ref="ABSLENGTH0xa8080f8"/>
     70502       </extra>
     70503     </material>


Story continues :doc:`/geant4/geometry/materials/material_properties`



2014/02/18 Geant4 Dybinst Rebuild
------------------------------------

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




Examining `dybinst-external` `dybinst-common.sh` note that geant4 is built 
with standard LCG Builders kicked off with `pkg_build geant4` which is defined
in `common.sh`


::

    291 # to build using LCG_Builders
    292 pkg_build () {
    293 
    294     pkg=$1 ; shift
    295     goto $SITEROOT/lcgcmt/LCG_Builders/$pkg/cmt
    296     cmt config
    297     source setup.sh
    298 
    299     #cmt_macro LCG_BuildPolicy LCG_tardir LCG_Builders 
    300     #cmt_macro LCG_BuildPolicy LCG_builddir LCG_Builders 
    301 
    302     echo "LCG_tardir=\"$LCG_tardir\""
    303     if [ ! -d ${LCG_tardir} ] ; then
    304     mkdir -p ${LCG_tardir}
    305     fi
    306     echo "LCG_builddir=\"$LCG_builddir\""
    307     if [ ! -d ${LCG_builddir} ] ; then
    308     mkdir -p ${LCG_builddir}
    309     fi
    310 
    311     cmt config
    312     if [ -r setup.sh ] ; then
    313     source setup.sh
    314     else
    315     echo "Failed to setup $gluedir"
    316     exit 1
    317     fi
    318 
    319     #echo "## begin env dump ##"
    320     #env
    321     #echo "## end env dump ##"
    322     #echo "## begin cmt dump ##"
    323     #cmt show tags
    324     #cmt show macros
    325     #echo "## end cmt dump ##"
    326     #cmt show macro LCG_basesystem
    327 
    328     for cmd in get config make install
    329     do
    330     echo "$pkg: running \"cmt pkg_$cmd\""
    331     cmt pkg_$cmd
    332     check_cmd
    333     done
    334 
    335     goback
    336 }



The cmt pkg_install invokes::

    [blyth@belle7 lib]$ cat /data1/env/local/dyb/NuWa-trunk/lcgcmt/LCG_Builders/geant4/scripts/geant4_install.sh
    #!/bin/sh

    # . ${LCG_BUILDPOLICYROOT_DIR}/scripts/common.sh

    echo "geant4: installing code"
    cd ${G4INSTALL}
    mkdir -p ${LCG_destdir}
    for dir in lib include
    do
        target="${LCG_destdir}/$dir"
        if [ -d "$target" ] ; then
            echo "geant4: $target already exists, remove to force reinstall"
        else
            tar -cf - $dir | (cd ${LCG_destdir} && tar -xf -)
        fi
    done
    cd ${LCG_destdir}/lib

    # curious move contents of Linux-g++ one up and remove Linux-g++  
    if [ -d "${G4SYSTEM}" ] ; then
        mv ${G4SYSTEM}/* .
        rmdir ${G4SYSTEM}
    fi

    ... then data downloading 



Installation checks for `G4INSTALL/lib` and `G4INSTALL/include`::

    [blyth@belle7 lib]$ echo $G4INSTALL
    /data1/env/local/dyb/NuWa-trunk/../external/build/LCG/geant4.9.2.p01
    [blyth@belle7 lib]$ 
    [blyth@belle7 lib]$ ll  $G4INSTALL/
    total 324
    -rw-r--r--  1 blyth blyth   4029 Mar 16  2009 LICENSE
    -rwxr-xr-x  1 blyth blyth 142639 Mar 16  2009 Configure
    drwxr-xr-x  2 blyth blyth   4096 Mar 16  2009 ReleaseNotes
    drwxr-xr-x  4 blyth blyth   4096 Mar 16  2009 environments
    -rw-rw-r--  1 blyth blyth      0 Feb 16  2011 .geant4.9.2.p01.patch2
    -rw-rw-r--  1 blyth blyth      0 Feb 16  2011 .geant4.9.2.p01.patch
    drwxrwxr-x  3 blyth blyth   4096 Feb 16  2011 lib
    drwxrwxr-x 34 blyth blyth   4096 Sep 18 18:32 ..
    -rw-rw-r--  1 blyth blyth      0 Sep 18 18:44 .geant4.9.2.p01.patch3
    drwxr-xr-x 11 blyth blyth   4096 Sep 18 19:17 .
    drwxr-xr-x 22 blyth blyth   4096 Sep 18 19:21 source
    drwxr-xr-x  5 blyth blyth   4096 Oct  1 19:40 examples
    drwxrwxr-x  3 blyth blyth   4096 Oct  1 20:06 bin
    drwxrwxr-x  6 blyth blyth   4096 Oct  2 20:13 tmp
    drwxr-xr-x  4 blyth blyth   4096 Dec  4 15:13 config
    drwxrwxr-x  2 blyth blyth 135168 Feb 19 14:01 include
    [blyth@belle7 lib]$ 
     
And propagates from there to `LCG_destdir`::

    [blyth@belle7 lib]$ echo ${LCG_destdir}
    /data1/env/local/dyb/NuWa-trunk/../external/geant4/4.9.2.p01/i686-slc5-gcc41-dbg
    [blyth@belle7 lib]$ ll  ${LCG_destdir}/
    total 196
    drwxrwxr-x 3 blyth blyth   4096 Feb 16  2011 ..
    drwxrwxr-x 2 blyth blyth   4096 Feb 16  2011 lib.prior
    drwxrwxr-x 5 blyth blyth   4096 Sep 18 19:55 .
    drwxrwxr-x 2 blyth blyth 180224 Nov 14 18:36 include
    drwxrwxr-x 2 blyth blyth   4096 Feb 19 14:34 lib


As this is kinda expensive do this manually::

    [blyth@belle7 4.9.2.p01]$ pwd
    /data1/env/local/dyb/external/geant4/4.9.2.p01
    [blyth@belle7 4.9.2.p01]$ mv i686-slc5-gcc41-dbg i686-slc5-gcc41-dbg.prior 



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



