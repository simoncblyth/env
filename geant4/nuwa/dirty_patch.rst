Make sense of dirty patch
===========================

.. contents:: :local:

Problems
---------

* Find that existing NuWa geant4 patch already modified files that I need to change (the MPV ones).
* the patch show signs of being **dirty** (presumably hand editing to add an older patch)

Moral
------

The moral of this mess is **NEVER USE PATCHES**, always stuff code that you are 
using into a repository and clone from there.

Reproducing a patch
--------------------

Hmm means must first reproduce the preexisting geant4 patches::

    cd /data1/env/local/dyb/external/build/LCG

    [blyth@belle7 LCG]$ mkdir g4checkpatch
    [blyth@belle7 LCG]$ tar zxf geant4.9.2.p01.tar.gz -C g4checkpatch
    [blyth@belle7 LCG]$ l g4checkpatch/
    total 4
    drwxr-xr-x 7 blyth blyth 4096 Mar 16  2009 geant4.9.2.p01
    [blyth@belle7 LCG]$ mv g4checkpatch/geant4.9.2.p01 g4checkpatch/geant4.9.2.p01.orig
    [blyth@belle7 LCG]$ cp -R geant4.9.2.p01 g4checkpatch/



::

    [blyth@belle7 g4checkpatch]$ diff -r --brief geant4.9.2.p01.orig geant4.9.2.p01
    Only in geant4.9.2.p01: bin
    Only in geant4.9.2.p01: .geant4.9.2.p01.patch
    Only in geant4.9.2.p01: .geant4.9.2.p01.patch2
    Only in geant4.9.2.p01: include
    Only in geant4.9.2.p01: lib
    Files geant4.9.2.p01.orig/source/digits_hits/utils/src/G4ScoreLogColorMap.cc and geant4.9.2.p01/source/digits_hits/utils/src/G4ScoreLogColorMap.cc differ
    Files geant4.9.2.p01.orig/source/digits_hits/utils/src/G4VScoreColorMap.cc and geant4.9.2.p01/source/digits_hits/utils/src/G4VScoreColorMap.cc differ
    Files geant4.9.2.p01.orig/source/geometry/solids/Boolean/src/G4SubtractionSolid.cc and geant4.9.2.p01/source/geometry/solids/Boolean/src/G4SubtractionSolid.cc differ
    Files geant4.9.2.p01.orig/source/materials/include/G4MaterialPropertyVector.hh and geant4.9.2.p01/source/materials/include/G4MaterialPropertyVector.hh differ
    Files geant4.9.2.p01.orig/source/materials/src/G4MaterialPropertiesTable.cc and geant4.9.2.p01/source/materials/src/G4MaterialPropertiesTable.cc differ
    Files geant4.9.2.p01.orig/source/materials/src/G4MaterialPropertyVector.cc and geant4.9.2.p01/source/materials/src/G4MaterialPropertyVector.cc differ
    Files geant4.9.2.p01.orig/source/processes/electromagnetic/lowenergy/src/G4hLowEnergyLoss.cc and geant4.9.2.p01/source/processes/electromagnetic/lowenergy/src/G4hLowEnergyLoss.cc differ
    Files geant4.9.2.p01.orig/source/processes/hadronic/processes/include/G4ElectronNuclearProcess.hh and geant4.9.2.p01/source/processes/hadronic/processes/include/G4ElectronNuclearProcess.hh differ
    Files geant4.9.2.p01.orig/source/processes/hadronic/processes/include/G4PhotoNuclearProcess.hh and geant4.9.2.p01/source/processes/hadronic/processes/include/G4PhotoNuclearProcess.hh differ
    Files geant4.9.2.p01.orig/source/processes/hadronic/processes/include/G4PositronNuclearProcess.hh and geant4.9.2.p01/source/processes/hadronic/processes/include/G4PositronNuclearProcess.hh differ
    Files geant4.9.2.p01.orig/source/processes/hadronic/processes/src/G4ElectronNuclearProcess.cc and geant4.9.2.p01/source/processes/hadronic/processes/src/G4ElectronNuclearProcess.cc differ
    Files geant4.9.2.p01.orig/source/processes/hadronic/processes/src/G4PhotoNuclearProcess.cc and geant4.9.2.p01/source/processes/hadronic/processes/src/G4PhotoNuclearProcess.cc differ
    Files geant4.9.2.p01.orig/source/processes/optical/include/G4OpBoundaryProcess.hh and geant4.9.2.p01/source/processes/optical/include/G4OpBoundaryProcess.hh differ
    Only in geant4.9.2.p01/source/processes/optical/include: G4OpBoundaryProcess.hh.orig
    Files geant4.9.2.p01.orig/source/visualization/HepRep/include/cheprep/DeflateOutputStreamBuffer.h and geant4.9.2.p01/source/visualization/HepRep/include/cheprep/DeflateOutputStreamBuffer.h differ
    Only in geant4.9.2.p01: tmp

    [blyth@belle7 g4checkpatch]$ cd geant4.9.2.p01
    [blyth@belle7 geant4.9.2.p01]$ rm -rf bin .geant4.9.2.p01.patch .geant4.9.2.p01.patch2 include lib tmp G4OpBoundaryProcess.hh.orig

    ## after cleaning the detritus

    [blyth@belle7 g4checkpatch]$ diff -r --brief geant4.9.2.p01.orig geant4.9.2.p01
    Files geant4.9.2.p01.orig/source/digits_hits/utils/src/G4ScoreLogColorMap.cc and geant4.9.2.p01/source/digits_hits/utils/src/G4ScoreLogColorMap.cc differ
    Files geant4.9.2.p01.orig/source/digits_hits/utils/src/G4VScoreColorMap.cc and geant4.9.2.p01/source/digits_hits/utils/src/G4VScoreColorMap.cc differ
    Files geant4.9.2.p01.orig/source/geometry/solids/Boolean/src/G4SubtractionSolid.cc and geant4.9.2.p01/source/geometry/solids/Boolean/src/G4SubtractionSolid.cc differ
    Files geant4.9.2.p01.orig/source/materials/include/G4MaterialPropertyVector.hh and geant4.9.2.p01/source/materials/include/G4MaterialPropertyVector.hh differ
    Files geant4.9.2.p01.orig/source/materials/src/G4MaterialPropertiesTable.cc and geant4.9.2.p01/source/materials/src/G4MaterialPropertiesTable.cc differ
    Files geant4.9.2.p01.orig/source/materials/src/G4MaterialPropertyVector.cc and geant4.9.2.p01/source/materials/src/G4MaterialPropertyVector.cc differ
    Files geant4.9.2.p01.orig/source/processes/electromagnetic/lowenergy/src/G4hLowEnergyLoss.cc and geant4.9.2.p01/source/processes/electromagnetic/lowenergy/src/G4hLowEnergyLoss.cc differ
    Files geant4.9.2.p01.orig/source/processes/hadronic/processes/include/G4ElectronNuclearProcess.hh and geant4.9.2.p01/source/processes/hadronic/processes/include/G4ElectronNuclearProcess.hh differ
    Files geant4.9.2.p01.orig/source/processes/hadronic/processes/include/G4PhotoNuclearProcess.hh and geant4.9.2.p01/source/processes/hadronic/processes/include/G4PhotoNuclearProcess.hh differ
    Files geant4.9.2.p01.orig/source/processes/hadronic/processes/include/G4PositronNuclearProcess.hh and geant4.9.2.p01/source/processes/hadronic/processes/include/G4PositronNuclearProcess.hh differ
    Files geant4.9.2.p01.orig/source/processes/hadronic/processes/src/G4ElectronNuclearProcess.cc and geant4.9.2.p01/source/processes/hadronic/processes/src/G4ElectronNuclearProcess.cc differ
    Files geant4.9.2.p01.orig/source/processes/hadronic/processes/src/G4PhotoNuclearProcess.cc and geant4.9.2.p01/source/processes/hadronic/processes/src/G4PhotoNuclearProcess.cc differ
    Files geant4.9.2.p01.orig/source/processes/optical/include/G4OpBoundaryProcess.hh and geant4.9.2.p01/source/processes/optical/include/G4OpBoundaryProcess.hh differ
    Files geant4.9.2.p01.orig/source/visualization/HepRep/include/cheprep/DeflateOutputStreamBuffer.h and geant4.9.2.p01/source/visualization/HepRep/include/cheprep/DeflateOutputStreamBuffer.h differ
    [blyth@belle7 g4checkpatch]$ 

    [blyth@belle7 g4checkpatch]$ diff -u -r geant4.9.2.p01.orig geant4.9.2.p01 > geant4.9.2.p01.patch0     


Path inconsistency in the patch makes me suspect hand editing of patch files.
Even after removing the 2nd small patch changes, cannot establish 
a match due to different diff ordering.


Extend diff.py to dump single file patches
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* :env:`base/diff.py`

::

        [blyth@belle7 g4checkpatch]$ diff.py geant4.9.2.p01.orig geant4.9.2.p01 -p
        [blyth@belle7 g4checkpatch]$ l patches/
        total 92
        -rw-rw-r-- 1 blyth blyth  388 Mar  4 20:15 geant4.9.2.p01_environments_g4py_config_module.gmk.patch
        -rw-rw-r-- 1 blyth blyth  418 Mar  4 20:15 geant4.9.2.p01_environments_g4py_configure.patch
        -rw-rw-r-- 1 blyth blyth  384 Mar  4 20:15 geant4.9.2.p01_source_digits_hits_utils_src_G4ScoreLogColorMap.cc.patch
        -rw-rw-r-- 1 blyth blyth  407 Mar  4 20:15 geant4.9.2.p01_source_digits_hits_utils_src_G4VScoreColorMap.cc.patch
        -rw-rw-r-- 1 blyth blyth 1230 Mar  4 20:15 geant4.9.2.p01_source_geometry_solids_Boolean_src_G4SubtractionSolid.cc.patch
        -rw-rw-r-- 1 blyth blyth  680 Mar  4 20:15 geant4.9.2.p01_source_materials_include_G4MaterialPropertiesTable.hh.patch
        -rw-rw-r-- 1 blyth blyth  888 Mar  4 20:15 geant4.9.2.p01_source_materials_include_G4MaterialPropertyVector.hh.patch
        -rw-rw-r-- 1 blyth blyth  934 Mar  4 20:15 geant4.9.2.p01_source_materials_src_G4MaterialPropertiesTable.cc.patch
        -rw-rw-r-- 1 blyth blyth 4080 Mar  4 20:15 geant4.9.2.p01_source_materials_src_G4MaterialPropertyVector.cc.patch
        -rw-rw-r-- 1 blyth blyth  429 Mar  4 20:15 geant4.9.2.p01_source_persistency_gdml_include_G4GDMLWrite.hh.patch
        -rw-rw-r-- 1 blyth blyth 2346 Mar  4 20:15 geant4.9.2.p01_source_persistency_gdml_src_G4GDMLWrite.cc.patch
        -rw-rw-r-- 1 blyth blyth 1517 Mar  4 20:15 geant4.9.2.p01_source_processes_electromagnetic_lowenergy_src_G4hLowEnergyLoss.cc.patch
        -rw-rw-r-- 1 blyth blyth  393 Mar  4 20:15 geant4.9.2.p01_source_processes_hadronic_processes_include_G4ElectronNuclearProcess.hh.patch
        -rw-rw-r-- 1 blyth blyth  383 Mar  4 20:15 geant4.9.2.p01_source_processes_hadronic_processes_include_G4PhotoNuclearProcess.hh.patch
        -rw-rw-r-- 1 blyth blyth  898 Mar  4 20:15 geant4.9.2.p01_source_processes_hadronic_processes_include_G4PositronNuclearProcess.hh.patch
        -rw-rw-r-- 1 blyth blyth  765 Mar  4 20:15 geant4.9.2.p01_source_processes_hadronic_processes_src_G4ElectronNuclearProcess.cc.patch
        -rw-rw-r-- 1 blyth blyth  737 Mar  4 20:15 geant4.9.2.p01_source_processes_hadronic_processes_src_G4PhotoNuclearProcess.cc.patch
        -rw-rw-r-- 1 blyth blyth 1307 Mar  4 20:15 geant4.9.2.p01_source_processes_optical_include_G4OpBoundaryProcess.hh.patch
        -rw-rw-r-- 1 blyth blyth  406 Mar  4 20:15 geant4.9.2.p01_source_visualization_HepRep_include_cheprep_DeflateOutputStreamBuffer.h.patch
        -rw-rw-r-- 1 blyth blyth  666 Mar  4 20:15 geant4.9.2.p01_source_visualization_VRML_GNUmakefile.patch
        -rw-rw-r-- 1 blyth blyth  663 Mar  4 20:15 geant4.9.2.p01_source_visualization_VRML_include_G4VRML2FileSceneHandler.hh.patch
        -rw-rw-r-- 1 blyth blyth 1634 Mar  4 20:15 geant4.9.2.p01_source_visualization_VRML_src_G4VRML2FileSceneHandler.cc.patch
        -rw-rw-r-- 1 blyth blyth  711 Mar  4 20:15 geant4.9.2.p01_source_visualization_VRML_src_G4VRML2SceneHandlerFunc.icc.patch


Create `patch-;patch-match` 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    [blyth@belle7 g4checkpatch]$ t patch-match
    patch-match is a function
    patch-match () 
    { 
        local line;
        local mpatch=$1;
        local opatch=$mpatch.$FUNCNAME;
        local msg="=== $FUNCNAME :";
        echo $msg truncating $opatch : reconstructing a combi patch from single file patches in matched order;
        echo -n >$opatch;
        local fpatch;
        grep -- +++ $mpatch | while read line; do
            fpatch=$(patch-match-line $line);
            echo $fpatch;
            [ -f "$fpatch" ] && cat $fpatch >>$opatch;
            [ ! -f "$fpatch" ] && echo $msg WARNING : FAILED to find $fpatch;
        done
    }
    [blyth@belle7 g4checkpatch]$ t patch-match-dirty
    patch-match-dirty is a function
    patch-match-dirty () 
    { 
        type $FUNCNAME;
        local msg="=== $FUNCNAME :";
        local patch="geant4.9.2.p01.patch";
        patch-match $patch;
        echo scrubbing the omitted line in the dirty patch;
        perl -pi -e 's,diff -u -r geant4.9.2.p01.orig/source/processes/optical/include/G4OpBoundaryProcess.hh geant4.9.2.p01/source/processes/optical/include/G4OpBoundaryProcess.hh\n,,' $patch.patch-match;
        local cmd="diff $patch $patch.patch-match";
        echo $msg $cmd;
        eval $cmd;
        echo $msg $cmd
    }

::

    [blyth@belle7 patches]$ pwd
    /data1/env/local/dyb/external/build/LCG/g4checkpatch/patches

    [blyth@belle7 patches]$ patch-;patch-match geant4.9.2.p01.patch
    === patch-match : truncating geant4.9.2.p01.patch.patch-match : reconstructing a combi patch from single file patches in matched order
    geant4.9.2.p01_source_geometry_solids_Boolean_src_G4SubtractionSolid.cc.patch
    geant4.9.2.p01_source_processes_electromagnetic_lowenergy_src_G4hLowEnergyLoss.cc.patch
    geant4.9.2.p01_source_processes_hadronic_processes_include_G4ElectronNuclearProcess.hh.patch
    geant4.9.2.p01_source_processes_hadronic_processes_include_G4PhotoNuclearProcess.hh.patch
    geant4.9.2.p01_source_processes_hadronic_processes_include_G4PositronNuclearProcess.hh.patch
    geant4.9.2.p01_source_processes_hadronic_processes_src_G4ElectronNuclearProcess.cc.patch
    geant4.9.2.p01_source_processes_hadronic_processes_src_G4PhotoNuclearProcess.cc.patch
    geant4.9.1.p01_source_processes_optical_include_G4OpBoundaryProcess.hh.patch
    geant4.9.2.p01_source_materials_include_G4MaterialPropertyVector.hh.patch
    geant4.9.2.p01_source_materials_src_G4MaterialPropertiesTable.cc.patch
    geant4.9.2.p01_source_materials_src_G4MaterialPropertyVector.cc.patch

Huh this is the new dyb, without my mods... how this diff between the order matched patches ?
This is due to the inconsistent name geant4.9.1.p01 vs geant4.9.2.p01 preventing inclusion of
the new patch::

    [blyth@belle7 patches]$ diff geant4.9.2.p01.patch geant4.9.2.p01.patch.patch-match 
    ...
    < diff -u -r geant4.9.2.p01.orig/source/processes/hadronic/processes/src/G4PhotoNuclearProcess.cc geant4.9.2.p01/source/processes/hadronic/processes/src/G4PhotoNuclearProcess.cc
    < --- geant4.9.2.p01.orig/source/processes/hadronic/processes/src/G4PhotoNuclearProcess.cc      2009-03-16 10:06:45.000000000 -0400
    < +++ geant4.9.2.p01/source/processes/hadronic/processes/src/G4PhotoNuclearProcess.cc   2009-06-11 01:22:12.000000000 -0400
    ---
    > --- geant4.9.2.p01.orig/source/processes/hadronic/processes/src/G4PhotoNuclearProcess.cc      2009-03-16 22:06:45.000000000 +0800
    > +++ geant4.9.2.p01/source/processes/hadronic/processes/src/G4PhotoNuclearProcess.cc   2014-03-04 18:04:57.000000000 +0800
    168,200c162,163
    < --- geant4.9.1.p01/source/processes/optical/include/G4OpBoundaryProcess.hh    2007-10-15 17:16:24.000000000 -0400
    < +++ geant4.9.1.p01/source/processes/optical/include/G4OpBoundaryProcess.hh    2009-04-14 11:28:42.000000000 -0400
    < @@ -408,10 +408,23 @@
    <  void G4OpBoundaryProcess::DoReflection()
    <  {
    <          if ( theStatus == LambertianReflection ) {
    < -
    < -          NewMomentum = G4LambertianRand(theGlobalNormal);
    < -          theFacetNormal = (NewMomentum - OldMomentum).unit();
    < -
    < +       // wangzhe
    < +       // Original:
    < +          //NewMomentum = G4LambertianRand(theGlobalNormal);
    < +          //theFacetNormal = (NewMomentum - OldMomentum).unit();
    < +       
    < +       // Temp Fix:
    < +       if(theGlobalNormal.mag()==0) {
    < +            std::cout<<"Error. Zero caught. A normal vector with mag be 0. May trigger a infinite loop later."<<std::endl;
    < +            std::cout<<"A temporary solution: Photon is forced to go back along its original path."<<std::endl;
    < +            std::cout<<"Test from MDC09a tells the effect of this bug is tiny."<<std::endl;
    < +         G4ThreeVector myVec(0,0,0);
    < +         theFacetNormal = (myVec - OldMomentum).unit();
    < +       } else {
    < +         NewMomentum = G4LambertianRand(theGlobalNormal);
    < +         theFacetNormal = (NewMomentum - OldMomentum).unit();
    < +       }
    < +       // wz
    <          }
    <          else if ( theFinish == ground ) {
    <  


After kludging to handle the inconsistencies, get down to diff-of-diffs with **patch-match-dirty**
as below with just date differences and offset hunk start line numbers for G4OpBoundaryProcess.hh::

    > +++ geant4.9.2.p01/source/processes/hadronic/processes/src/G4ElectronNuclearProcess.cc        2014-03-04 18:04:57.000000000 +0800
    150,151c150,151
    < --- geant4.9.2.p01.orig/source/processes/hadronic/processes/src/G4PhotoNuclearProcess.cc      2009-03-16 10:06:45.000000000 -0400
    < +++ geant4.9.2.p01/source/processes/hadronic/processes/src/G4PhotoNuclearProcess.cc   2009-06-11 01:22:12.000000000 -0400
    ---
    > --- geant4.9.2.p01.orig/source/processes/hadronic/processes/src/G4PhotoNuclearProcess.cc      2009-03-16 22:06:45.000000000 +0800
    > +++ geant4.9.2.p01/source/processes/hadronic/processes/src/G4PhotoNuclearProcess.cc   2014-03-04 18:04:57.000000000 +0800
    168,170c168,170
    < --- geant4.9.1.p01/source/processes/optical/include/G4OpBoundaryProcess.hh    2007-10-15 17:16:24.000000000 -0400
    < +++ geant4.9.1.p01/source/processes/optical/include/G4OpBoundaryProcess.hh    2009-04-14 11:28:42.000000000 -0400
    < @@ -408,10 +408,23 @@
    ---
    > --- geant4.9.2.p01.orig/source/processes/optical/include/G4OpBoundaryProcess.hh       2009-03-16 22:06:47.000000000 +0800
    > +++ geant4.9.2.p01/source/processes/optical/include/G4OpBoundaryProcess.hh    2014-03-04 18:04:58.000000000 +0800
    > @@ -299,10 +299,23 @@
    199,200c199,200
    < --- geant4.9.2.p01.orig/source/materials/include/G4MaterialPropertyVector.hh  2009-09-29 19:55:36.000000000 -0700
    < +++ geant4.9.2.p01/source/materials/include/G4MaterialPropertyVector.hh       2009-09-28 16:51:45.000000000 -0700
    ---
    > --- geant4.9.2.p01.orig/source/materials/include/G4MaterialPropertyVector.hh  2009-03-16 22:05:57.000000000 +0800
    > +++ geant4.9.2.p01/source/materials/include/G4MaterialPropertyVector.hh       2014-03-04 18:04:54.000000000 +0800
    220,221c220,221

Inconsistent hunk start line numbers::

    < @@ -408,10 +408,23 @@
    > @@ -299,10 +299,23 @@


It seems that the difference in hunk start line numbers does not matter for patch application ?
Checking /data1/env/local/dyb/external/build/LCG/geant4.9.2.p01/source/processes/optical/include/G4OpBoundaryProcess.hh
it appears to be correctly applied.

* http://www.gnu.org/software/diffutils/manual/html_node/Detailed-Unified.html#Detailed-Unified  
* http://stackoverflow.com/questions/10950412/what-does-1-1-mean-in-gits-diff-output

::

    298 inline
    299 void G4OpBoundaryProcess::DoReflection()
    300 {
    301         if ( theStatus == LambertianReflection ) {
    302       // wangzhe
    303       // Original:
    304           //NewMomentum = G4LambertianRand(theGlobalNormal);
    305           //theFacetNormal = (NewMomentum - OldMomentum).unit();
    306 
    307       // Temp Fix:
    308       if(theGlobalNormal.mag()==0) {
    309             std::cout<<"Error. Zero caught. A normal vector with mag be 0. May trigger a infinite loop later."<<std::endl;
    310             std::cout<<"A temporary solution: Photon is forced to go back along its original path."<<std::endl;
    311             std::cout<<"Test from MDC09a tells the effect of this bug is tiny."<<std::endl;
    312         G4ThreeVector myVec(0,0,0);
    313         theFacetNormal = (myVec - OldMomentum).unit();
    314       } else {
    315         NewMomentum = G4LambertianRand(theGlobalNormal);
    316         theFacetNormal = (NewMomentum - OldMomentum).unit();
    317       }
    318       // wz
    319         }
    320         else if ( theFinish == ground ) {
    321 


But there is a `G4OpBoundaryProcess.hh.orig` dropping in the source, presumably this is punishment from `patch`::

    [blyth@belle7 geant4.9.2.p01]$ pwd
    /data1/env/local/dyb/external/build/LCG/geant4.9.2.p01

    [blyth@belle7 geant4.9.2.p01]$ find . -name '*.orig' -exec ls -l {} \;
    -rw-r--r-- 1 blyth blyth 10395 Mar 16  2009 ./include/G4OpBoundaryProcess.hh.orig
    -rw-r--r-- 1 blyth blyth 10395 Mar 16  2009 ./source/processes/optical/include/G4OpBoundaryProcess.hh.orig
    [blyth@belle7 geant4.9.2.p01]$ 


Note the inconsistencies for G4OpBoundaryProcess 

* "geant4.9.1.p01" is present where "geant4.9.2.p01" should be
* the diff command line is omitted 

The patch that are comparing against was clearly fabricated in a dirty manner.


test patching
~~~~~~~~~~~~~~~

With the clean patch::

    [blyth@belle7 g4checkpatch]$ cp geant4.9.2.p01.orig/source/processes/optical/include/G4OpBoundaryProcess.hh G4OpBoundaryProcess.hh
    [blyth@belle7 g4checkpatch]$ patch -b G4OpBoundaryProcess.hh patches/geant4.9.2.p01_source_processes_optical_include_G4OpBoundaryProcess.hh.patch
    patching file G4OpBoundaryProcess.hh
    [blyth@belle7 g4checkpatch]$ ll G4Op*
    -rw-r--r-- 1 blyth blyth 10395 Mar  5 12:59 G4OpBoundaryProcess.hh.orig
    -rw-r--r-- 1 blyth blyth 11031 Mar  5 12:59 G4OpBoundaryProcess.hh

Split off the dirty hunk in the patch, compare dirty to clean patch::

    [blyth@belle7 g4checkpatch]$ diff geant4.9.2.p01_G4OpBoundaryProcess.hh.patch patches/geant4.9.2.p01_source_processes_optical_include_G4OpBoundaryProcess.hh.patch
    1,3c1,4
    < --- geant4.9.1.p01/source/processes/optical/include/G4OpBoundaryProcess.hh    2007-10-15 17:16:24.000000000 -0400
    < +++ geant4.9.1.p01/source/processes/optical/include/G4OpBoundaryProcess.hh    2009-04-14 11:28:42.000000000 -0400
    < @@ -408,10 +408,23 @@
    ---
    > diff -u -r geant4.9.2.p01.orig/source/processes/optical/include/G4OpBoundaryProcess.hh geant4.9.2.p01/source/processes/optical/include/G4OpBoundaryProcess.hh
    > --- geant4.9.2.p01.orig/source/processes/optical/include/G4OpBoundaryProcess.hh       2009-03-16 22:06:47.000000000 +0800
    > +++ geant4.9.2.p01/source/processes/optical/include/G4OpBoundaryProcess.hh    2014-03-04 18:04:58.000000000 +0800
    > @@ -299,10 +299,23 @@
    [blyth@belle7 g4checkpatch]$ 

Confirm that dirty patch succeeds with complaint and .orig dropping::

    [blyth@belle7 g4checkpatch]$ cp geant4.9.2.p01.orig/source/processes/optical/include/G4OpBoundaryProcess.hh .
    [blyth@belle7 g4checkpatch]$ patch -N G4OpBoundaryProcess.hh -i geant4.9.2.p01_G4OpBoundaryProcess.hh.patch
    patching file G4OpBoundaryProcess.hh
    Hunk #1 succeeded at 299 (offset -109 lines).

    [blyth@belle7 g4checkpatch]$ ll G4OpBoundaryProcess.hh*
    -rw-r--r-- 1 blyth blyth 10395 Mar  5 14:09 G4OpBoundaryProcess.hh.orig
    -rw-r--r-- 1 blyth blyth 11031 Mar  5 14:09 G4OpBoundaryProcess.hh


Four dirty patches
~~~~~~~~~~~~~~~~~~~~~

Check the log::

    [blyth@belle7 dyb]$ grep Hunk *.log
    dybinst-20140303-145546.log:Hunk #1 succeeded at 1330 with fuzz 1.
    dybinst-20140303-145546.log:Hunk #1 succeeded at 299 (offset -109 lines).
    dybinst-20140303-145546.log:Hunk #1 succeeded at 4659 (offset 44 lines).
    dybinst-20140303-145546.log:Hunk #1 succeeded at 27 with fuzz 2.
    [blyth@belle7 dyb]$ 

::

    161632 openscientistvis: "running command: patch -N -p0 -i /data1/env/local/dyb/NuWa-trunk/lcgcmt/LCG_Builders/openscientistvis/patches/osc_16_11.patch"
    161633 + patch -N -p0 -i /data1/env/local/dyb/NuWa-trunk/lcgcmt/LCG_Builders/openscientistvis/patches/osc_16_11.patch
    161634 patching file OnX/source/Gtk/GtkUI.cxx
    161635 Hunk #1 succeeded at 1330 with fuzz 1.
    ...
    206730 geant4: "running command: patch -N -p1 -i /data1/env/local/dyb/NuWa-trunk/lcgcmt/LCG_Builders/geant4/patches/geant4.9.2.p01.patch"
    206731 + patch -N -p1 -i /data1/env/local/dyb/NuWa-trunk/lcgcmt/LCG_Builders/geant4/patches/geant4.9.2.p01.patch
    206732 patching file source/geometry/solids/Boolean/src/G4SubtractionSolid.cc
    206733 patching file source/processes/electromagnetic/lowenergy/src/G4hLowEnergyLoss.cc
    206734 patching file source/processes/hadronic/processes/include/G4ElectronNuclearProcess.hh
    206735 patching file source/processes/hadronic/processes/include/G4PhotoNuclearProcess.hh
    206736 patching file source/processes/hadronic/processes/include/G4PositronNuclearProcess.hh
    206737 patching file source/processes/hadronic/processes/src/G4ElectronNuclearProcess.cc
    206738 patching file source/processes/hadronic/processes/src/G4PhotoNuclearProcess.cc
    206739 patching file source/processes/optical/include/G4OpBoundaryProcess.hh
    206740 Hunk #1 succeeded at 299 (offset -109 lines).
    206741 patching file source/materials/include/G4MaterialPropertyVector.hh
    206742 patching file source/materials/src/G4MaterialPropertiesTable.cc
    206743 patching file source/materials/src/G4MaterialPropertyVector.cc
    ...
    216214 + patch -N -p1 -i /data1/env/local/dyb/NuWa-trunk/lcgcmt/LCG_Builders/more/patches/more-0.8.2.patch
    216215 patching file more/Makefile.am
    216216 patching file more/Makefile.in
    216217 Hunk #1 succeeded at 4659 (offset 44 lines).
    ...
    216247 + patch -N -p1 -i /data1/env/local/dyb/NuWa-trunk/lcgcmt/LCG_Builders/more/patches/more-0.8.2-popen.patch
    216248 patching file more/io/ipstream.cc
    216249 Hunk #1 succeeded at 27 with fuzz 2.


There are more `.orig`
~~~~~~~~~~~~~~~~~~~~~~~~

::

    [blyth@belle7 external]$ find . -name '*.orig'
    ./build/LCG/g4checkpatch/G4OpBoundaryProcess.hh.orig
    ./build/LCG/g4checkpatch/clean/G4OpBoundaryProcess.hh.orig
    ./build/LCG/g4checkpatch/geant4.9.2.p01.orig
    ./build/LCG/OpenScientist/osc_vis/bin_obuild/osc_vis/16.11/Resources/foreign/Python.obuild.orig
    ./build/LCG/OpenScientist/osc_vis/bin_obuild/osc_vis/16.11/Resources/foreign/Motif.obuild.orig
    ./build/LCG/OpenScientist/OnX/source/Gtk/GtkUI.cxx.orig
    ./build/LCG/OpenScientist/foreign/Python.obuild.orig
    ./build/LCG/OpenScientist/foreign/Motif.obuild.orig
    ./build/LCG/more-0.8.3/more/io/ipstream.cc.orig
    ./build/LCG/more-0.8.3/more/Makefile.in.orig
    ./build/LCG/geant4.9.2.p01/include/G4OpBoundaryProcess.hh.orig
    ./build/LCG/geant4.9.2.p01/source/processes/optical/include/G4OpBoundaryProcess.hh.orig
    ./OpenScientist/16.11/i686-slc5-gcc41-dbg/osc_vis/16.11/Resources/foreign/Python.obuild.orig
    ./OpenScientist/16.11/i686-slc5-gcc41-dbg/osc_vis/16.11/Resources/foreign/Motif.obuild.orig
    ./OpenScientist/16.11/i686-slc5-gcc41-dbg/Resources/foreign/Python.obuild.orig
    ./OpenScientist/16.11/i686-slc5-gcc41-dbg/Resources/foreign/Motif.obuild.orig
    ./geant4/4.9.2.p01/i686-slc5-gcc41-dbg/include/G4OpBoundaryProcess.hh.orig
    [blyth@belle7 external]$ 


Check other geant4.9.2.p01.patch2
-----------------------------------

::

    [blyth@belle7 g4checkpatch]$ patch-           
    [blyth@belle7 g4checkpatch]$ patch-match-clean
    patch-match-clean is a function
    patch-match-clean () 
    { 
        type $FUNCNAME;
        local msg="=== $FUNCNAME :";
        local patch="patches/geant4.9.2.p01.patch2";
        patch-match $patch "---";
        echo $msg scrubbing all the diff cmdline in patch;
        perl -pi -e 's,^diff -u -r .*\n$,,' $patch.patch-match;
        patch-match-compare $patch
    }
    === patch-match-clean : mpatch patches/geant4.9.2.p01.patch2 patchdir patches marker ---
    === patch-match : truncating patches/geant4.9.2.p01.patch2.patch-match : reconstructing a combi patch from single file patches in matched order
    patches/geant4.9.2.p01_source_digits_hits_utils_src_G4ScoreLogColorMap.cc.patch
    patches/geant4.9.2.p01_source_visualization_HepRep_include_cheprep_DeflateOutputStreamBuffer.h.patch
    patches/geant4.9.2.p01_source_digits_hits_utils_src_G4VScoreColorMap.cc.patch
    === patch-match-clean : scrubbing all the diff cmdline in patch
    === patch-match-compare : diff patches/geant4.9.2.p01.patch2 patches/geant4.9.2.p01.patch2.patch-match
    1,2c1,2
    < --- geant4.9.2.p01/source/digits_hits/utils/src/G4ScoreLogColorMap.cc 2009-03-16 15:05:42.000000000 +0100
    < +++ geant4.9.2.p01.new/source/digits_hits/utils/src/G4ScoreLogColorMap.cc     2010-07-30 15:25:21.265772830 +0200
    ---
    > --- geant4.9.2.p01.orig/source/digits_hits/utils/src/G4ScoreLogColorMap.cc    2009-03-16 22:05:42.000000000 +0800
    > +++ geant4.9.2.p01/source/digits_hits/utils/src/G4ScoreLogColorMap.cc 2014-03-04 18:04:59.000000000 +0800
    12,13c12,13
    < --- geant4.9.2.p01/source/visualization/HepRep/include/cheprep/DeflateOutputStreamBuffer.h    2009-03-16 15:06:51.000000000 +0100
    < +++ geant4.9.2.p01.new/source/visualization/HepRep/include/cheprep/DeflateOutputStreamBuffer.h        2010-07-30 15:27:57.315772592 +0200
    ---
    > --- geant4.9.2.p01.orig/source/visualization/HepRep/include/cheprep/DeflateOutputStreamBuffer.h       2009-03-16 22:06:51.000000000 +0800
    > +++ geant4.9.2.p01/source/visualization/HepRep/include/cheprep/DeflateOutputStreamBuffer.h    2014-03-04 18:04:55.000000000 +0800
    22,23c22,23
    < --- geant4.9.2.p01/source/digits_hits/utils/src/G4VScoreColorMap.cc   2009-03-16 15:05:42.000000000 +0100
    < +++ geant4.9.2.p01.new/source/digits_hits/utils/src/G4VScoreColorMap.cc       2010-07-30 16:06:01.225772558 +0200
    ---
    > --- geant4.9.2.p01.orig/source/digits_hits/utils/src/G4VScoreColorMap.cc      2009-03-16 22:05:42.000000000 +0800
    > +++ geant4.9.2.p01/source/digits_hits/utils/src/G4VScoreColorMap.cc   2014-03-04 18:04:59.000000000 +0800
    === patch-match-compare : diff patches/geant4.9.2.p01.patch2 patches/geant4.9.2.p01.patch2.patch-match
    [blyth@belle7 g4checkpatch]$ 




My dyb.old Geant4 mods were ontop of two patches
--------------------------------------------------

Need to extracate::

    [blyth@belle7 g4checkpatch]$ pwd
    /data1/env/local/dyb.old/external/build/LCG/g4checkpatch

    [blyth@belle7 g4checkpatch]$ diff -r --brief geant4.9.2.p01.orig geant4.9.2.p01
    Files geant4.9.2.p01.orig/environments/g4py/config/module.gmk and geant4.9.2.p01/environments/g4py/config/module.gmk differ
    Files geant4.9.2.p01.orig/environments/g4py/configure and geant4.9.2.p01/environments/g4py/configure differ

            ## this was me experimenting with g4py, only to find that be GetPoly API I was intyerested in was not provided

    Files geant4.9.2.p01.orig/source/digits_hits/utils/src/G4ScoreLogColorMap.cc and geant4.9.2.p01/source/digits_hits/utils/src/G4ScoreLogColorMap.cc differ
    Files geant4.9.2.p01.orig/source/digits_hits/utils/src/G4VScoreColorMap.cc and geant4.9.2.p01/source/digits_hits/utils/src/G4VScoreColorMap.cc differ
    Files geant4.9.2.p01.orig/source/geometry/solids/Boolean/src/G4SubtractionSolid.cc and geant4.9.2.p01/source/geometry/solids/Boolean/src/G4SubtractionSolid.cc differ

           ## from the patches

    Files geant4.9.2.p01.orig/source/materials/include/G4MaterialPropertiesTable.hh and geant4.9.2.p01/source/materials/include/G4MaterialPropertiesTable.hh differ
    Only in geant4.9.2.p01/source/materials/include: G4MaterialPropertiesTable.hh.orig
    Files geant4.9.2.p01.orig/source/materials/include/G4MaterialPropertyVector.hh and geant4.9.2.p01/source/materials/include/G4MaterialPropertyVector.hh differ
    Only in geant4.9.2.p01/source/materials/include: G4MaterialPropertyVector.hh.orig
    Files geant4.9.2.p01.orig/source/materials/src/G4MaterialPropertiesTable.cc and geant4.9.2.p01/source/materials/src/G4MaterialPropertiesTable.cc differ
    Files geant4.9.2.p01.orig/source/materials/src/G4MaterialPropertyVector.cc and geant4.9.2.p01/source/materials/src/G4MaterialPropertyVector.cc differ
    Only in geant4.9.2.p01/source/materials/src: G4MaterialPropertyVector.cc.orig

           ## interference between my changes and patch


    Files geant4.9.2.p01.orig/source/persistency/gdml/include/G4GDMLWrite.hh and geant4.9.2.p01/source/persistency/gdml/include/G4GDMLWrite.hh differ

           ## buffer size limitation, truncating id fix

    Files geant4.9.2.p01.orig/source/persistency/gdml/src/G4GDMLWrite.cc and geant4.9.2.p01/source/persistency/gdml/src/G4GDMLWrite.cc differ

           ## buffer size, replacing hardcoded 99

    Files geant4.9.2.p01.orig/source/processes/electromagnetic/lowenergy/src/G4hLowEnergyLoss.cc and geant4.9.2.p01/source/processes/electromagnetic/lowenergy/src/G4hLowEnergyLoss.cc differ
    Files geant4.9.2.p01.orig/source/processes/hadronic/processes/include/G4ElectronNuclearProcess.hh and geant4.9.2.p01/source/processes/hadronic/processes/include/G4ElectronNuclearProcess.hh differ
    Files geant4.9.2.p01.orig/source/processes/hadronic/processes/include/G4PhotoNuclearProcess.hh and geant4.9.2.p01/source/processes/hadronic/processes/include/G4PhotoNuclearProcess.hh differ
    Files geant4.9.2.p01.orig/source/processes/hadronic/processes/include/G4PositronNuclearProcess.hh and geant4.9.2.p01/source/processes/hadronic/processes/include/G4PositronNuclearProcess.hh differ
    Files geant4.9.2.p01.orig/source/processes/hadronic/processes/src/G4ElectronNuclearProcess.cc and geant4.9.2.p01/source/processes/hadronic/processes/src/G4ElectronNuclearProcess.cc differ
    Files geant4.9.2.p01.orig/source/processes/hadronic/processes/src/G4PhotoNuclearProcess.cc and geant4.9.2.p01/source/processes/hadronic/processes/src/G4PhotoNuclearProcess.cc differ
    Files geant4.9.2.p01.orig/source/processes/optical/include/G4OpBoundaryProcess.hh and geant4.9.2.p01/source/processes/optical/include/G4OpBoundaryProcess.hh differ
    Only in geant4.9.2.p01/source/processes/optical/include: G4OpBoundaryProcess.hh.orig

    Files geant4.9.2.p01.orig/source/visualization/HepRep/include/cheprep/DeflateOutputStreamBuffer.h and geant4.9.2.p01/source/visualization/HepRep/include/cheprep/DeflateOutputStreamBuffer.h differ

    Files geant4.9.2.p01.orig/source/visualization/VRML/GNUmakefile and geant4.9.2.p01/source/visualization/VRML/GNUmakefile differ
        
         ## ill advised debug

    Files geant4.9.2.p01.orig/source/visualization/VRML/include/G4VRML2FileSceneHandler.hh and geant4.9.2.p01/source/visualization/VRML/include/G4VRML2FileSceneHandler.hh differ
    Files geant4.9.2.p01.orig/source/visualization/VRML/src/G4VRML2FileSceneHandler.cc and geant4.9.2.p01/source/visualization/VRML/src/G4VRML2FileSceneHandler.cc differ

         ## VRML2 precision fix + debug

    Files geant4.9.2.p01.orig/source/visualization/VRML/src/G4VRML2SceneHandlerFunc.icc and geant4.9.2.p01/source/visualization/VRML/src/G4VRML2SceneHandlerFunc.icc differ

         ##  commented debug




