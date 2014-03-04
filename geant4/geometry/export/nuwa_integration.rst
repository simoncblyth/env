Integration of Export Functionality into NuWa
================================================

Forking techniques
------------------ 

#. CMTEXTRATAGS geant4_with_gdml geant4_with_g4dae from override file
#. dybinst command line option, analogous to "-O" "-g" dybinst option that sets force_optdbg 

Given slave monopolization of the `~/.dybinstrc` override file and
the fact that only the first existing override file is adhered to, it is 
not convenient to configure a setting like the below in the override file::

   global_extra=geant4_with_gdml,geant4_with_g4dae

More convenient to do this with a dybinst command line option analogous
to switching on optimized building, this would allow multiple installs 
on the same node with geometry export capability enabled in one of them


CMT gymnastics
~~~~~~~~~~~~~~~~

Is a CMTEXTRATAGS addition the appropriate way to implement the fork.

* http://www.cmtsite.net/CMTDoc.html


Dependencies
-------------

#. GDML, Non-standardly built Geant4 libary libG4gdml.so 
#. G4DAE, Non-standard Geant4 library: libG4DAE.so
#. Non-standard external dependency of GDML and G4DAE: XercesC  


Geant4 GDML Building
---------------------

Tag '''geant4_with_gdml''' should cause building of libG4gdml.so 

* http://dayabay.ihep.ac.cn/tracs/dybsvn/changeset/22595



Geant4 Patches
---------------

Older Geant4 requires patches:

#. VRML2 precision fix, in order provide useful geometry cross-check
#. spilling the beans on material properties

Patching Geant4 
~~~~~~~~~~~~~~~~~

See these from /data1/env/local/dyb.old/NuWa-trunk for how to add Geant4 patches::
    
   lcgcmt/LCG_Builders/geant4/patches/geant4.9.2.p01.patch3   
   lcgcmt/LCG_Builders/geant4/scripts/geant4_config.sh


Geant4 Additions
-----------------

* G4DAE should be part of Geant4, but for now needs to live in its own repo
  whats the appropriate way to include it in the build

Hailing from before GDML was included with G4 ?

  * NuWa-trunk/lcgcmt/LCG_Interfaces/GDML/cmt/requirements



G4DAE builder or extended patch ?
-------------------------------------

A separate builder would need intimate access to Geant4 build::

    [blyth@belle7 LCG_Builders]$ grep Geant4  */cmt/requirements
    geant4/cmt/requirements:macro geant4_srcdir "geant${Geant4_config_version}"
    geant4/cmt/requirements:apply_pattern buildscripts_project_local destdir=${LCG_reldir}/geant4/${Geant4_config_version}/${LCG_CMTCONFIG}
    geant4/cmt/requirements:set LCG_tarfilename "geant${Geant4_config_version}.tar.gz"
    geant4/cmt/requirements:macro LCG_sourcefiles "geant${Geant4_config_version}.tar.gz"
    geant4/cmt/requirements:set G4INSTALL "${LCG_builddir}/geant$(Geant4_config_version)"
    openscientistvis/cmt/requirements:use Geant4          v* LCG_Interfaces
    [blyth@belle7 LCG_Builders]$ 
    [blyth@belle7 LCG_Builders]$ 
    [blyth@belle7 LCG_Builders]$ pwd
    /data1/env/local/dyb/NuWa-trunk/lcgcmt/LCG_Builders

The OSC scripts are complicated because of this

* openscientistvis/scripts/openscientistvis_make.sh


LCG Builder-ization
---------------------

Ape the geant4 build with ::

    fenv
    cd $SITEROOT/lcgcmt/LCG_Builders/geant4/cmt
    cmt config
    . setup.sh

    cmt pkg_get      # looks to not use the script
    cmt pkg_config   # could kludge svn checkout and patch GNUMakefile in here 
    cmt pkg_make
    cmt pkg_install

Considered a separate LCG_Builder for g4dae, but that 
makes things more complicated. It is essentially a
rather extended patch against Geant4.


Arghh existing patch touches MPV
---------------------------------

* /data1/env/local/dyb/NuWa-trunk/lcgcmt/LCG_Builders/geant4/patches/geant4.9.2.p01.patch

Hmm means must first reproduce this patch...
::

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



Modifications
--------------

lhcb
~~~~~~

::

    Performing status on external item at 'lhcb'
    M       lhcb/Sim/GaussTools/cmt/requirements
    A  +    lhcb/Sim/GaussTools/src/Components/GiGaRunActionGDML.cpp
    A  +    lhcb/Sim/GaussTools/src/Components/GiGaRunActionGDML.h
    M       lhcb/Sim/GiGa/cmt/requirements


This seems too low level. Create G4DAE interface package and use that perhaps.::

    [blyth@belle7 lhcb]$ svn diff Sim/GaussTools/cmt/requirements
    Index: Sim/GaussTools/cmt/requirements
    ===================================================================
    --- Sim/GaussTools/cmt/requirements     (revision 22589)
    +++ Sim/GaussTools/cmt/requirements     (working copy)
    @@ -31,6 +31,11 @@
     apply_pattern     component_library library=GaussTools
     apply_pattern     linker_library    library=GaussToolsLib
     
    +# SCB : enable GDML,DAE,WRL export by GiGaRunActionGDML
    +macro_append GaussTools_cppflags " -DEXPORT_G4GDML=1 -DEXPORT_G4DAE=1 -DEXPORT_G4WRL=1 "
    +macro_append GaussTools_linkopts " -lG4DAE "
    +
    +
     # special linking with minimal G4RunManager to build genConf (necessary due
     # to G4 User Actions requiring it to exist and have physic list assigned to it)
     #============================================================================


This somehow seems wrong, the geant4 use with the appropriate tags
should bring along the appropiate dependencies like XercesC.::

    [blyth@belle7 lhcb]$ svn diff Sim/GiGa/cmt/requirements
    Index: Sim/GiGa/cmt/requirements
    ===================================================================
    --- Sim/GiGa/cmt/requirements   (revision 22589)
    +++ Sim/GiGa/cmt/requirements   (working copy)
    @@ -18,8 +18,15 @@
     use              GaudiAlg     v* 
     macro geant4_use "G4readout    v* Geant4" \
           dayabay   "Geant4      v* LCG_Interfaces"
    +
    +macro geant4_optional_use "" \
    +      geant4_with_gdml "XercesC v* LCG_Interfaces" 
    +
     use $(geant4_use)
     
    +use $(geant4_optional_use)
    +
    +


dybgaudi
~~~~~~~~

::

    Performing status on external item at 'dybgaudi'
    M       dybgaudi/Simulation/G4DataHelpers/cmt/requirements



installation
~~~~~~~~~~~~~~

Settings like switching on GDML need to be global    
as it impacts the geant4 build and all dependencies of geant4.

Initially tried a technique coming out of `~/.dybinstrc` but
thats not convenient for cohabiting dybinstalls, so plump
for greenfield dybinst option `./dybinst -X geant4_with_gdml trunk all` 
That stresses the need for the greenfield build.

* http://dayabay.ihep.ac.cn/tracs/dybsvn/changeset/22610



