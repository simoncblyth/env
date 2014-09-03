Integration of Export Functionality into NuWa
================================================

Commits
--------

* http://dayabay.ihep.ac.cn/tracs/dybsvn/changeset/22635
* http://dayabay.ihep.ac.cn/tracs/dybsvn/changeset/22636


Patches
---------

::

    [blyth@belle7 patches]$ pwd
    /data1/env/local/dybx/NuWa-trunk/lcgcmt/LCG_Builders/geant4/patches
    [blyth@belle7 patches]$ svn log -l10 -v
    ------------------------------------------------------------------------
    r22636 | blyth | 2014-03-05 20:23:17 +0800 (Wed, 05 Mar 2014) | 1 line
    Changed paths:
       M /lcgcmt/trunk/LCG_Builders/geant4/cmt/requirements
       M /lcgcmt/trunk/LCG_Builders/geant4/patches/geant4.9.2.p01.patch.patch2_source_persistency_GNUmakefile.patch
       M /lcgcmt/trunk/LCG_Builders/geant4/scripts/geant4_config.sh

    minor: change to tag geant4_with_dae to be closer to geant4 conventions 
    ------------------------------------------------------------------------
    r22635 | blyth | 2014-03-05 19:51:25 +0800 (Wed, 05 Mar 2014) | 1 line
    Changed paths:
       A /lcgcmt/trunk/LCG_Builders/geant4/patches/geant4.9.2.p01.patch.patch2_source_materials_include_G4MaterialPropertiesTable.hh.patch
       A /lcgcmt/trunk/LCG_Builders/geant4/patches/geant4.9.2.p01.patch.patch2_source_materials_include_G4MaterialPropertyVector.hh.patch
       A /lcgcmt/trunk/LCG_Builders/geant4/patches/geant4.9.2.p01.patch.patch2_source_materials_src_G4MaterialPropertyVector.cc.patch
       A /lcgcmt/trunk/LCG_Builders/geant4/patches/geant4.9.2.p01.patch.patch2_source_persistency_GNUmakefile.patch
       A /lcgcmt/trunk/LCG_Builders/geant4/patches/geant4.9.2.p01.patch.patch2_source_persistency_gdml_include_G4GDMLWrite.hh.patch
       A /lcgcmt/trunk/LCG_Builders/geant4/patches/geant4.9.2.p01.patch.patch2_source_persistency_gdml_src_G4GDMLWrite.cc.patch
       A /lcgcmt/trunk/LCG_Builders/geant4/patches/geant4.9.2.p01.patch.patch2_source_visualization_VRML_src_G4VRML2FileSceneHandler.cc.patch
       M /lcgcmt/trunk/LCG_Builders/geant4/scripts/geant4_config.sh

    minor: geant4 patches needed to integrate building G4DAE geometry export as COLLADA DAE documents functionaity, NOT ENABLED BY DEFAULT, requires CMTEXTRATAGS to include '''geant4_with_g4dae'''. notes at e/geant4/nuwa/create_incremental_patch 








Forking techniques
------------------ 

#. CMTEXTRATAGS geant4_with_gdml geant4_with_dae from override file
#. dybinst command line option, analogous to "-O" "-g" dybinst option that sets force_optdbg 

Given slave monopolization of the `~/.dybinstrc` override file and
the fact that only the first existing override file is adhered to, it is 
not convenient to configure a setting like the below in the override file::

   global_extra=geant4_with_gdml,geant4_with_dae

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

See :doc:`/geant4/nuwa/dirty_patch` 

Modifications
--------------

::

   ./dybinst -X geant4_with_dae trunk projects relax gaudi    # untouched ? 

   ./dybinst -X geant4_with_dae trunk projects lhcb           # needs CMT attention
   ./dybinst -X geant4_with_dae trunk projects dybgaudi


relax
~~~~~~~~

Not used ?::

    [blyth@belle7 relax]$ find . -name requirements -exec grep -H eant4 {} \;
    ./Dictionaries/GeantFourRflx/v8r0p01/cmt/requirements:macro Geant4_native_version "8.0.p01"
    ./Dictionaries/GeantFourRflx/v8r0p01/cmt/requirements:macro Geant4__8_0_p01__Rflx_use_linkopts " -L$(Geant4_home)/lib                              \
    ./Dictionaries/GeantFourRflx/v8r0p01/cmt/requirements:apply_pattern relax_dictionary dictionary=Geant4__8_0_p01__             \
    ./Dictionaries/GeantFourRflx/v8r0p01/cmt/requirements:                               headerfiles=$(GEANTFOURRFLXROOT)/dict/Geant4Dict.h      \
    ./Dictionaries/GeantFourRflx/v9r0p01/cmt/requirements:macro Geant4_native_version "9.0.p01"
    ./Dictionaries/GeantFourRflx/v9r0p01/cmt/requirements:macro Geant4__9_0_p01__Rflx_use_linkopts " -L$(Geant4_home)/lib                              \
    ./Dictionaries/GeantFourRflx/v9r0p01/cmt/requirements:apply_pattern relax_dictionary dictionary=Geant4__9_0_p01__             \
    ./Dictionaries/GeantFourRflx/v9r0p01/cmt/requirements:                               headerfiles=$(V9R0P01ROOT)/dict/Geant4Dict.h      \
    ./Dictionaries/GeantFourRflx/v7r1p01a/cmt/requirements:macro Geant4_native_version "7.1.p01a"
    ./Dictionaries/GeantFourRflx/v7r1p01a/cmt/requirements:macro Geant4__7_1_p01a__Rflx_use_linkopts " -L$(Geant4_home)/lib                              \
    ./Dictionaries/GeantFourRflx/v7r1p01a/cmt/requirements:apply_pattern relax_dictionary dictionary=Geant4__7_1_p01a__             \
    ./Dictionaries/GeantFourRflx/v7r1p01a/cmt/requirements:                               headerfiles=$(GEANTFOURRFLXROOT)/dict/Geant4Dict.h      \
    ./LCG_Interfaces/GeantFour/cmt/requirements:package Geant4
    ./LCG_Interfaces/GeantFour/cmt/requirements:macro Geant4_native_version __SPECIFY_MACRO__>>Geant4_native_version<<
    ./LCG_Interfaces/GeantFour/cmt/requirements:macro Geant4_home "$(LCG_external)/geant4/$(Geant4_native_version)/$(LCG_system)"
    ./LCG_Interfaces/GeantFour/cmt/requirements:include_dirs $(Geant4_home)/share/include
    ./LCG_Interfaces/GeantFour/cmt/requirements:macro Geant4_linkopts "-L$(Geant4_home)/lib "        \
    ./LCG_Interfaces/GeantFour/cmt/requirements:      WIN32           "/LIBPATH:$(Geant4_home)/lib "
    [blyth@belle7 relax]$ 



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



geant4 liblist 
~~~~~~~~~~~~~~~~

Maybe adding library to geant4 means need some liblist action ?


docs
^^^^^^

* http://geant4.web.cern.ch/geant4/UserDocumentation/UsersGuides/InstallationGuide/BackupVersions/V9.4/html/ch02s03.html

At this point, you may choose one of two ways to compile and install the kernel
libraries, depending on your needs and system resources. From
$G4INSTALL/source::

   make

This will make one library for each "leaf" category (maximum library
granularity) and automatically produce a map of library use and dependencies.::

    make global

This will make global libraries, one for each major category.

The main advantage of the first approach is the speed of building the libraries
and of the application, which in some cases can be improved by a factor of two
or three compared to the "global library" approach.

Using the "granular library" approach a fairly large number (roughly 90) of
"leaf" libraries is produced. However, the dependencies and linking list are
evaluated and generated automatically on the fly. The top-level GNUmakefile in
$G4INSTALL/source parses the dependency files of Geant4 and produces a file
libname.map in $G4LIB. libname.map is produced by the tool liblist, whose
source code is in $G4INSTALL/config.

When building a binary application the script binmake.gmk in $G4INSTALL/config
will parse the user's dependency files and use libname.map to determine through
liblist the required libraries to add to the linking list. Only the required
libraries will be loaded in the link command.

The command make libmap issued from $G4INSTALL/source, allows manual rebuilding
of the dependency map. The command is issued by default in the normal build
process for granular libraries.

It is possible to install both "granular" and "compound" libraries, by typing
"make" and "make global" in sequence. In this case, to choose usage of granular
libraries at link time one should set the flag G4LIB_USE_GRANULAR in the
environment; otherwise compound libraries will be adopted by default.

libname.map
^^^^^^^^^^^^

Looks appropriate::

     27 G4DAE: G4volumes G4globman G4geometrymng G4geomdivision G4csg G4specsolids G4graphics_reps G4geomBoolean G4hepnumerics G4materials
     28 source/persistency/dae/GNUmakefile
     29 G4gdml: G4geometrymng G4globman G4geomdivision G4volumes G4csg G4specsolids G4graphics_reps G4geomBoolean G4hepnumerics G4materials
     30 source/persistency/gdml/GNUmakefile





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


export_all test in dybx installation
---------------------------------------

::

    [blyth@belle7 ~]$ nuwa-;DYB=x nuwa-setup 
    Creating setup scripts.
    Creating cleanup scripts.
    [blyth@belle7 cmt]$ cd ~/e/geant4/geometry/export
    [blyth@belle7 export]$ which nuwa.py
    /data1/env/local/dybx/NuWa-trunk/dybgaudi/InstallArea/scripts/nuwa.py

    [blyth@belle7 export]$ G4DAE_EXPORT_SEQUENCE=DVGX nuwa.py -n1 -m export_all
    ...
    GiGaRunActionExport::BeginOfRunAction i 0 c D
    FreeFilePath  return ./g4_01.dae i 2
    GiGaRunActionExport::WriteDAE to ./g4_01.dae recreatePoly 0
    G4DAEWrite::Write addPointerToName 1 recreatePoly 0 nodeindex -1
    G4DAE: Writing './g4_01.dae'...
    G4DAE: Writing asset metadata...
    G4DAE: Writing library_effects...
    G4DAE: Writing library_geometries...
    G4DAE: Writing library_materials...
    G4DAE: Writing structure/library_nodes...
    G4DAE: Writing library_visual_scenes...
    G4DAE::GetBorderSurface ... /dd/Geometry/Sites/lvNearHallTop#pvNearTopCover[1000]
    G4DAE::GetBorderSurface ... /dd/Geometry/RPC/lvRPCGasgap14#pvStrip14Array#pvStrip14ArrayOne:1#pvStrip14Unit[1]
    G4DAE::GetBorderSurface ... /dd/Geometry/RPC/lvRPCGasgap14#pvStrip14Array#pvStrip14ArrayOne:2#pvStrip14Unit[2]
    G4DAE::GetBorderSurface ... /dd/Geometry/RPC/lvRPCGasgap14#pvStrip14Array#pvStrip14ArrayOne:3#pvStrip14Unit[3]
    G4DAE::GetBorderSurface ... /dd/Geometry/RPC/lvRPCGasgap14#pvStrip14Array#pvStrip14ArrayOne:4#pvStrip14Unit[4]
    G4DAE::GetBorderSurface ... /dd/Geometry/RPC/lvRPCGasgap14#pvStrip14Array#pvStrip14ArrayOne:5#pvStrip14Unit[5]
    G4DAE::GetBorderSurface ... /dd/Geometry/RPC/lvRPCGasgap14#pvStrip14Array#pvStrip14ArrayOne:6#pvStrip14Unit[6]
    G4DAE::GetBorderSurface ... /dd/Geometry/RPC/lvRPCGasgap14#pvStrip14Array#pvStrip14ArrayOne:7#pvStrip14Unit[7]
    G4DAE::GetBorderSurface ... /dd/Geometry/RPC/lvRPCGasgap14#pvStrip14Array#pvStrip14ArrayOne:8#pvStrip14Unit[8]
    G4DAE::GetBorderSurface ... /dd/Geometry/RPC/lvRPCBarCham14#pvRPCGasgap14[1000]
    G4DAE::GetBorderSurface ... /dd/Geometry/RPC/lvRPCFoam#pvBarCham14Array#pvBarCham14ArrayOne:1#pvBarCham14Unit[1]
    G4DAE::GetBorderSurface ... /dd/Geometry/RPC/lvRPCFoam#pvBarCham14Array#pvBarCham14ArrayOne:2#pvBarCham14Unit[2]
    G4DAE::GetBorderSurface ... /dd/Geometry/RPC/lvRPCGasgap23#pvStrip23Array#pvStrip23ArrayOne:1#pvStrip23Unit[1]
    G4DAE::GetBorderSurface ... /dd/Geometry/RPC/lvRPCGasgap23#pvStrip23Array#pvStrip23ArrayOne:2#pvStrip23Unit[2]
    G4DAE::GetBorderSurface ... /dd/Geometry/RPC/lvRPCGasgap23#pvStrip23Array#pvStrip23ArrayOne:3#pvStrip23Unit[3]
    G4DAE::GetBorderSurface ... /dd/Geometry/RPC/lvRPCGasgap23#pvStrip23Array#pvStrip23ArrayOne:4#pvStrip23Unit[4]
    G4DAE::GetBorderSurface ... /dd/Geometry/RPC/lvRPCGasgap23#pvStrip23Array#pvStrip23ArrayOne:5#pvStrip23Unit[5]
    G4DAE::GetBorderSurface ... /dd/Geometry/RPC/lvRPCGasgap23#pvStrip23Array#pvStrip23ArrayOne:6#pvStrip23Unit[6]
    G4DAE::GetBorderSurface ... /dd/Geometry/RPC/lvRPCGasgap23#pvStrip23Array#pvStrip23ArrayOne:7#pvStrip23Unit[7]
    G4DAE::GetBorderSurface ... /dd/Geometry/RPC/lvRPCGasgap23#pvStrip23Array#pvStrip23ArrayOne:8#pvStrip23Unit[8]
    G4DAE::GetBorderSurface ... /dd/Geometry/RPC/lvRPCBarCham23#pvRPCGasgap23[1000]
    G4DAE::GetBorderSurface ... /dd/Geometry/RPC/lvRPCFoam#pvBarCham23Array#pvBarCham23ArrayOne:1#pvBarCham23Unit[1]
    G4DAE::GetBorderSurface ... /dd/Geometry/RPC/lvRPCFoam#pvBarCham23Array#pvBarCham23ArrayOne:2#pvBarCham23Unit[2]
    G4DAE::GetBorderSurface ... /dd/Geometry/RPC/lvRPCMod#pvRPCFoam[1000]
    G4DAE::GetBorderSurface ... /dd/Geometry/Sites/lvNearHallTop#pvNearTeleRpc#pvNearTeleRpc:1[1]
    G4DAE::GetBorderSurface ... /dd/Geometry/Sites/lvNearHallTop#pvNearTeleRpc#pvNearTeleRpc:2[2]
    G4DAE::GetBorderSurface ... /dd/Geometry/RPC/lvNearRPCRoof#pvNearUnSlopModArray#pvNearUnSlopModOne:1#pvNearUnSlopMod:1#pvNearSlopModUnit[1]









do nothing test in vanilla dyb installation
----------------------------------------------

It appears to succeed to do nothing, and was not too noisy, but may be some missing cleanup::

    [blyth@belle7 ~]$ fenv
    [blyth@belle7 ~]$ cd ~/e/geant4/geometry/export
    [blyth@belle7 export]$ 
    [blyth@belle7 export]$ 
    [blyth@belle7 export]$ which nuwa.py 
    /data1/env/local/dyb/NuWa-trunk/dybgaudi/InstallArea/scripts/nuwa.py
    [blyth@belle7 export]$ nuwa.py -n1 -m export_all
    ...
    GiGa                                  INFO Used  Event Action Object is GiGaEventActionSequence/GiGa.EventSeq
    GiGa                                  INFO Used  Run Action Object is GiGaRunActionExport/GiGa.GiGaRunActionExport
    ...
    GiGaRunActionExport::BeginOfRunAction i 0 c V
    GiGaRunActionExport::BeginOfRunAction i 1 c G
    FreeFilePath  return ./g4_01.gdml i 2
    GiGaRunActionExport::BeginOfRunAction i 2 c D
    FreeFilePath  return ./g4_01.dae i 2
    Start Run processing.
    DsPmtModel checking if applicable to opticalphoton

    Warning: G4MaterialPropertyVector::GetProperty
    ==> attempt to Retrieve Property above range

    Warning: G4MaterialPropertyVector::GetProperty
    ==> attempt to Retrieve Property above range

    Warning: G4MaterialPropertyVector::GetProperty
    ==> attempt to Retrieve Property above range

    Warning: G4MaterialPropertyVector::GetProperty
    ==> attempt to Retrieve Property above range

    Warning: G4MaterialPropertyVector::GetProperty
    ==> attempt to Retrieve Property above range

    Warning: G4MaterialPropertyVector::GetProperty
    ==> attempt to Retrieve Property above range

    Warning: G4MaterialPropertyVector::GetProperty
    ==> attempt to Retrieve Property above range

    Warning: G4MaterialPropertyVector::GetProperty
    ==> attempt to Retrieve Property above range

    Warning: G4MaterialPropertyVector::GetProperty
    ==> attempt to Retrieve Property above range

    Warning: G4MaterialPropertyVector::GetProperty
    ==> attempt to Retrieve Property above range

    Warning: G4MaterialPropertyVector::GetProperty
    ==> attempt to Retrieve Property above range

    Warning: G4MaterialPropertyVector::GetProperty
    ==> attempt to Retrieve Property above range

     G4: Number of events processed : 1 Timer: User=2389.91s Real=2730.12s Sys=0.51s
    ApplicationMgr                        INFO Application Manager Stopped successfully
    GiGaGeo                            SUCCESS  Exceptions/Errors/Warnings statistics:  0/0/2
    GiGaGeo                            SUCCESS  #WARNINGS  = 2 Message=' g4LVolume() is the obsolete method, use volume()!'
    GiGaGeo                            SUCCESS  #WARNINGS  = 1 Message='world():: Magnetic Field is not requested to be loaded '
    ToolSvc                               INFO Removing all tools created by ToolSvc
    GiGaGeo.DsPmtSensDet                  INFO DsPmtSensDet finalize
    GiGaGeo.DsRpcSensDet                  INFO DsRpcSensDet finalize
    GiGa.GiGaVis                          INFO GiGaVisManager:: finalize(): Delete the virualization manager
    Graphics systems deleted.
    Visualization Manager deleting...
    GiGa.GiGaMgr                          INFO GiGaRunManager:: GiGaRunManager Finalization
    ToolSvc.SequencerTimerTool            INFO ------------------------------------------------------------------------------------------------
    ToolSvc.SequencerTimerTool            INFO This machine has a speed about   2.21 times the speed of a 2.8 GHz Xeon.
    ToolSvc.SequencerTimerTool            INFO Algorithm          (millisec) |    <user> |   <clock> |      min       max | entries | total (s) |
    ToolSvc.SequencerTimerTool            INFO ------------------------------------------------------------------------------------------------
    ToolSvc.SequencerTimerTool            INFO ------------------------------------------------------------------------------------------------
    *****Chrono*****                      INFO ****************************************************************************************************
    *****Chrono*****                      INFO  The Final CPU consumption ( Chrono ) Table (ordered)
    *****Chrono*****                      INFO ****************************************************************************************************
    GiGa.GiGaMgr::processTheEvent()       INFO Time User   : Tot= 39.8[min]                                             #=  1
    ChronoStatSvc                         INFO Time User   : Tot= 41.9[min]                                             #=  1
    *****Chrono*****                      INFO ****************************************************************************************************
    ******Stat******                      INFO ****************************************************************************************************
    ******Stat******                      INFO  The Final stat Table (ordered)
    ******Stat******                      INFO ****************************************************************************************************
    ******Stat******                      INFO      Counter     |     #     |    sum     | mean/eff^* | rms/err^*  |     min     |     max     |
    ******Stat******                      INFO  "GiGaGeo:Warnin |         0 |          0 |     0.0000 |     0.0000 | 1.7977e+308 |-1.7977e+308 |
    ******Stat******                      INFO ****************************************************************************************************
    ChronoStatSvc.finalize()              INFO  Service finalized succesfully 
    ApplicationMgr                        INFO Application Manager Finalized successfully
    ApplicationMgr                        INFO Application Manager Terminated successfully
    Number of objects of type: GiGaRunManager created, but not destroyed:1
    Number of objects of type: GiGaPhysConstructorBase created, but not destroyed:3
    Number of objects of type: GiGaPhysicsConstructorBase created, but not destroyed:3
    Number of objects of type: GiGaPhysicsListBase created, but not destroyed:1
    Number of objects of type: GiGaPhysListBase created, but not destroyed:1
    Number of objects of type: GiGaSensDetBase created, but not destroyed:2
    GaudiTool       WARNING   Create/Destroy      (mis)balance 'DsPhysConsEM/GiGa.GiGaPhysListModular.DsPhysConsEM' Counts = 1
    GaudiTool       WARNING   Create/Destroy      (mis)balance 'DsPhysConsGeneral/GiGa.GiGaPhysListModular.DsPhysConsGeneral' Counts = 1
    GaudiTool       WARNING   Create/Destroy      (mis)balance 'DsPhysConsOptical/GiGa.GiGaPhysListModular.DsPhysConsOptical' Counts = 1
    GaudiTool       WARNING   Create/Destroy      (mis)balance 'DsPmtSensDet/GiGaGeo.DsPmtSensDet' Counts = 1
    GaudiTool       WARNING   Create/Destroy      (mis)balance 'DsRpcSensDet/GiGaGeo.DsRpcSensDet' Counts = 1
    GaudiTool       WARNING   Create/Destroy      (mis)balance 'GiGaPhysListModular/GiGa.GiGaPhysListModular' Counts = 1
    GaudiTool       WARNING   Create/Destroy      (mis)balance 'GiGaRunManager/GiGa.GiGaMgr' Counts = 1
    [blyth@belle7 export]$ 




Attach to see what the process is up to, propagating photons mostly::

    494           operator[](size_type __n) const
    (gdb) bt
    #0  0x041f811a in std::vector<G4NavigationLevel, std::allocator<G4NavigationLevel> >::operator[] (this=0xc4045f4, __n=12) at /usr/lib/gcc/i386-redhat-linux/4.1.2/../../../../include/c++/4.1.2/bits/stl_vector.h:494
    #1  0x041f81a3 in G4NavigationHistory::GetTopTransform (this=0xc4045f4) at /data1/env/local/dyb/NuWa-trunk/../external/build/LCG/geant4.9.2.p01/source/geometry/volumes/include/G4NavigationHistory.icc:102
    #2  0x0703aa3c in G4Navigator::ComputeLocalAxis (this=0xc4045e8, pVec=@0xbfd17220) at include/G4Navigator.icc:57
    #3  0x070365cb in G4Navigator::ComputeStep (this=0xc4045e8, pGlobalpoint=@0xbfd17208, pDirection=@0xbfd17220, pCurrentProposedStepLength=47809528.913293302, pNewSafety=@0xbfd17238) at src/G4Navigator.cc:628
    #4  0x04e096fa in G4Transportation::AlongStepGetPhysicalInteractionLength (this=0xc06d4e8, track=@0x10a5a5c8, currentMinimumStep=47809528.913293302, currentSafety=@0xbfd173b8, selection=0xc4042fc) at src/G4Transportation.cc:225
    #5  0x06e23e1b in G4VProcess::AlongStepGPIL (this=0xc06d4e8, track=@0x10a5a5c8, previousStepSize=17.522238749144233, currentMinimumStep=47809528.913293302, proposedSafety=@0xbfd173b8, selection=0xc4042fc)
        at /data1/env/local/dyb/NuWa-trunk/../external/build/LCG/geant4.9.2.p01/source/processes/management/include/G4VProcess.hh:447
    #6  0x06e22849 in G4SteppingManager::DefinePhysicalStepLength (this=0xc4041f0) at src/G4SteppingManager2.cc:235
    #7  0x06e1ee2c in G4SteppingManager::Stepping (this=0xc4041f0) at src/G4SteppingManager.cc:181
    #8  0x06e2d50a in G4TrackingManager::ProcessOneTrack (this=0xc4041c8, apValueG4Track=0x10a5a5c8) at src/G4TrackingManager.cc:126
    #9  0x06ea024f in G4EventManager::DoProcessing (this=0xc4039d8, anEvent=0x102ccca8) at src/G4EventManager.cc:185
    #10 0x06ea09e6 in G4EventManager::ProcessOneEvent (this=0xc4039d8, anEvent=0x102ccca8) at src/G4EventManager.cc:335
    #11 0xb4d2b5e8 in GiGaRunManager::processTheEvent (this=0xc403170) at ../src/component/GiGaRunManager.cpp:207
    #12 0xb4d2a522 in GiGaRunManager::retrieveTheEvent (this=0xc403170, event=@0xbfd17cf8) at ../src/component/GiGaRunManager.cpp:158
    #13 0xb4d0664f in GiGa::retrieveTheEvent (this=0xc402778, event=@0xbfd17cf8) at ../src/component/GiGa.cpp:469
    #14 0xb4d03564 in GiGa::operator>> (this=0xc402778, event=@0xbfd17cf8) at ../src/component/GiGaIGiGaSvc.cpp:73
    #15 0xb4d012fa in GiGa::retrieveEvent (this=0xc402778, event=@0xbfd17cf8) at ../src/component/GiGaIGiGaSvc.cpp:211
    #16 0xb4f4acd3 in DsPullEvent::execute (this=0xc3f5d00) at ../src/DsPullEvent.cc:54
    #17 0x069c1408 in Algorithm::sysExecute (this=0xc3f5d00) at ../src/Lib/Algorithm.cpp:558
    #18 0x0350ed4e in DybBaseAlg::sysExecute (this=0xc3f5d00) at ../src/lib/DybBaseAlg.cc:53
    #19 0x02cc6fd4 in GaudiSequencer::execute (this=0xbeb8140) at ../src/lib/GaudiSequencer.cpp:100
    #20 0x069c1408 in Algorithm::sysExecute (this=0xbeb8140) at ../src/Lib/Algorithm.cpp:558
    #21 0x02c5e68f in GaudiAlgorithm::sysExecute (this=0xbeb8140) at ../src/lib/GaudiAlgorithm.cpp:161
    #22 0x06a3d41a in MinimalEventLoopMgr::executeEvent (this=0xba77900) at ../src/Lib/MinimalEventLoopMgr.cpp:450
    #23 0x038ba956 in DybEventLoopMgr::executeEvent (this=0xba77900, par=0x0) at ../src/DybEventLoopMgr.cpp:125
    #24 0x038bb18a in DybEventLoopMgr::nextEvent (this=0xba77900, maxevt=1) at ../src/DybEventLoopMgr.cpp:188
    #25 0x06a3bdbd in MinimalEventLoopMgr::executeRun (this=0xba77900, maxevt=1) at ../src/Lib/MinimalEventLoopMgr.cpp:400
    #26 0x093096d9 in ApplicationMgr::executeRun (this=0xb744aa0, evtmax=1) at ../src/ApplicationMgr/ApplicationMgr.cpp:867
    #27 0x0829bf57 in method_3426 (retaddr=0xc4f7d00, o=0xb744ecc, arg=@0xb7b0c20) at ../i686-slc5-gcc41-dbg/dict/GaudiKernel/dictionary_dict.cpp:4375
    #28 0x001d6add in ROOT::Cintex::Method_stub_with_context (context=0xb7b0c18, result=0xc53d26c, libp=0xc53d2c4) at cint/cintex/src/CINTFunctional.cxx:319
    #29 0x0330e034 in ?? ()
    #30 0x0b7b0c18 in ?? ()
    #31 0x0c53d26c in ?? ()
    #32 0x00000000 in ?? ()
    Current language:  auto; currently c++
    (gdb) 

Continuing.::

    Program received signal SIGINT, Interrupt.
    [Switching to Thread -1208088368 (LWP 13703)]
    __gnu_cxx::__normal_iterator<G4MPVEntry* const*, std::vector<G4MPVEntry*, std::allocator<G4MPVEntry*> > >::base (this=0xbfd171d0) at /usr/lib/gcc/i386-redhat-linux/4.1.2/../../../../include/c++/4.1.2/bits/stl_iterator.h:715
    715           base() const
    (gdb) bt
    #0  __gnu_cxx::__normal_iterator<G4MPVEntry* const*, std::vector<G4MPVEntry*, std::allocator<G4MPVEntry*> > >::base (this=0xbfd171d0) at /usr/lib/gcc/i386-redhat-linux/4.1.2/../../../../include/c++/4.1.2/bits/stl_iterator.h:715
    #1  0xb6063b4e in __gnu_cxx::operator-<G4MPVEntry* const*, G4MPVEntry* const*, std::vector<G4MPVEntry*, std::allocator<G4MPVEntry*> > > (__lhs=@0xbfd171d0, __rhs=@0xbfd171cc)
        at /usr/lib/gcc/i386-redhat-linux/4.1.2/../../../../include/c++/4.1.2/bits/stl_iterator.h:809
    #2  0xb6063bca in std::vector<G4MPVEntry*, std::allocator<G4MPVEntry*> >::size (this=0xc2aba50) at /usr/lib/gcc/i386-redhat-linux/4.1.2/../../../../include/c++/4.1.2/bits/stl_vector.h:402
    #3  0xb6062518 in G4MaterialPropertyVector::GetAdjacentBins (this=0xc2aba50, aPhotonEnergy=3.1289435029520031e-06, left=0xbfd17244, right=0xbfd17240) at src/G4MaterialPropertyVector.cc:395
    #4  0xb6062be9 in G4MaterialPropertyVector::GetProperty (this=0xc2aba50, aPhotonEnergy=3.1289435029520031e-06) at src/G4MaterialPropertyVector.cc:225
    #5  0x04701884 in G4OpAbsorption::GetMeanFreePath (this=0xc337b70, aTrack=@0x10ee0068) at src/G4OpAbsorption.cc:139
    #6  0x04d5357a in G4VDiscreteProcess::PostStepGetPhysicalInteractionLength (this=0xc337b70, track=@0x10ee0068, previousStepSize=0, condition=0xc4042f8)
        at /data1/env/local/dyb/NuWa-trunk/../external/build/LCG/geant4.9.2.p01/source/processes/management/include/G4VDiscreteProcess.hh:152
    #7  0x06e23e95 in G4VProcess::PostStepGPIL (this=0xc337b70, track=@0x10ee0068, previousStepSize=0, condition=0xc4042f8) at /data1/env/local/dyb/NuWa-trunk/../external/build/LCG/geant4.9.2.p01/source/processes/management/include/G4VProcess.hh:464
    #8  0x06e2255a in G4SteppingManager::DefinePhysicalStepLength (this=0xc4041f0) at src/G4SteppingManager2.cc:165
    #9  0x06e1ee2c in G4SteppingManager::Stepping (this=0xc4041f0) at src/G4SteppingManager.cc:181
    #10 0x06e2d50a in G4TrackingManager::ProcessOneTrack (this=0xc4041c8, apValueG4Track=0x10ee0068) at src/G4TrackingManager.cc:126
    #11 0x06ea024f in G4EventManager::DoProcessing (this=0xc4039d8, anEvent=0x102ccca8) at src/G4EventManager.cc:185
    #12 0x06ea09e6 in G4EventManager::ProcessOneEvent (this=0xc4039d8, anEvent=0x102ccca8) at src/G4EventManager.cc:335
    #13 0xb4d2b5e8 in GiGaRunManager::processTheEvent (this=0xc403170) at ../src/component/GiGaRunManager.cpp:207
    #14 0xb4d2a522 in GiGaRunManager::retrieveTheEvent (this=0xc403170, event=@0xbfd17cf8) at ../src/component/GiGaRunManager.cpp:158
    #15 0xb4d0664f in GiGa::retrieveTheEvent (this=0xc402778, event=@0xbfd17cf8) at ../src/component/GiGa.cpp:469
    #16 0xb4d03564 in GiGa::operator>> (this=0xc402778, event=@0xbfd17cf8) at ../src/component/GiGaIGiGaSvc.cpp:73
    #17 0xb4d012fa in GiGa::retrieveEvent (this=0xc402778, event=@0xbfd17cf8) at ../src/component/GiGaIGiGaSvc.cpp:211
    #18 0xb4f4acd3 in DsPullEvent::execute (this=0xc3f5d00) at ../src/DsPullEvent.cc:54
    #19 0x069c1408 in Algorithm::sysExecute (this=0xc3f5d00) at ../src/Lib/Algorithm.cpp:558
    #20 0x0350ed4e in DybBaseAlg::sysExecute (this=0xc3f5d00) at ../src/lib/DybBaseAlg.cc:53
    #21 0x02cc6fd4 in GaudiSequencer::execute (this=0xbeb8140) at ../src/lib/GaudiSequencer.cpp:100
    #22 0x069c1408 in Algorithm::sysExecute (this=0xbeb8140) at ../src/Lib/Algorithm.cpp:558
    #23 0x02c5e68f in GaudiAlgorithm::sysExecute (this=0xbeb8140) at ../src/lib/GaudiAlgorithm.cpp:161
    #24 0x06a3d41a in MinimalEventLoopMgr::executeEvent (this=0xba77900) at ../src/Lib/MinimalEventLoopMgr.cpp:450
    #25 0x038ba956 in DybEventLoopMgr::executeEvent (this=0xba77900, par=0x0) at ../src/DybEventLoopMgr.cpp:125
    #26 0x038bb18a in DybEventLoopMgr::nextEvent (this=0xba77900, maxevt=1) at ../src/DybEventLoopMgr.cpp:188
    #27 0x06a3bdbd in MinimalEventLoopMgr::executeRun (this=0xba77900, maxevt=1) at ../src/Lib/MinimalEventLoopMgr.cpp:400
    #28 0x093096d9 in ApplicationMgr::executeRun (this=0xb744aa0, evtmax=1) at ../src/ApplicationMgr/ApplicationMgr.cpp:867
    #29 0x0829bf57 in method_3426 (retaddr=0xc4f7d00, o=0xb744ecc, arg=@0xb7b0c20) at ../i686-slc5-gcc41-dbg/dict/GaudiKernel/dictionary_dict.cpp:4375
    #30 0x001d6add in ROOT::Cintex::Method_stub_with_context (context=0xb7b0c18, result=0xc53d26c, libp=0xc53d2c4) at cint/cintex/src/CINTFunctional.cxx:319
    #31 0x0330e034 in ?? ()
    #32 0x0b7b0c18 in ?? ()
    #33 0x0c53d26c in ?? ()
    #34 0x00000000 in ?? ()
    (gdb) c



nuwa.py integration
---------------------

It would be nice to do export with a simple::

     nuwa.py -X 

BUT, below attempted implementation approach fails as need to setup a generator 
as done in `~/e/geant4/geometry/export/export_all.py`.
But thats too specific a thing to do inside nuwa.py. 

Perhaps using an earlier hook than **GiGaRunAction** can avoid this requirement.
Need to understand GiGa details to make progress with this.

* http://lhcb-comp.web.cern.ch/lhcb-comp/Frameworks/Gaudi/Documents/GiGa.pdf


::

    [blyth@belle7 DybPython]$ svn st 
    M       python/DybPython/cmdline.py
    M       python/DybPython/Control.py
    [blyth@belle7 DybPython]$ svn diff
    Index: python/DybPython/cmdline.py
    ===================================================================
    --- python/DybPython/cmdline.py (revision 22643)
    +++ python/DybPython/cmdline.py (working copy)
    @@ -134,6 +134,9 @@
         parser.add_argument("-G", "--detector",default= "",
                           help="Specify a non-default, top-level geometry file")
     
    +    parser.add_argument("-X", "--export",action="store_true", default=False,
    +                      help="Export Geant4 detector geometry in DAE,GDML,WRL formats. Requires special build with geant4_with_dae tag.")
    +
         parser.add_argument("-K", "--leak-check-execute",action="store_true",default= False,
                           help="Use Hephaestus memory tracker")
         parser.add_argument("--leak-check-method",action="append",
    Index: python/DybPython/Control.py
    ===================================================================
    --- python/DybPython/Control.py (revision 22643)
    +++ python/DybPython/Control.py (working copy)
    @@ -606,7 +606,59 @@
                 XmlDetDesc.Configure()
             return
     
    +    def configure_geometry_export(self):
    +        """
    +        When the `-X` or `--export` option is provided 
    +        to nuwa.py this configures geometry exporting 
    +        by GiGaRunActionExport at the begin of run.  
     
    +        For standard dybinstalls this is expected to run without error
    +        but perform no exports.
    +
    +        For special dybinstalls with the geant4_with_dae CMTEXTRATAGS
    +        this will export the Geant4 geometry into multiple files 
    +        in formats: 
    +
    +        DAE
    +            Standard 3D XML file format (COLLADA) with material and 
    +            surface properties as a function of wavelength 
    +            in metadata "extra" tags.  Geometry scenegraph is retained. 
    +        GDML
    +             Geant4 "native" persistency format 
    +        WRL
    +             Primitive VRML2 3D file format (form the 1990s) 
    +             with flattened scenegraph
    +
    +        The details are controlled by 2 envvars:
    +
    +        G4DAE_EXPORT_DIR
    +                Directory into which the geometry files are writted, default is "."
    +        G4DAE_EXPORT_SEQUENCE
    +                String controlling the export order, eg with "DGV" 
    +                for export sequence: DAE, GDML, WRL
    + 
    +        """
    +        if not self.opts.export:
    +            log.info("SKIPPING geometry export setup  ")
    +            return
    +
    +        log.info("configure geometry export via GiGaRunActionExport")
    +        from GiGa.GiGaConf import GiGa, GiGaRunManager
    +        giga = GiGa("GiGa")
    +        gigarm = GiGaRunManager("GiGa.GiGaMgr")
    +        gigarm.Verbosity = 2  # skip dumping particle properties
    +
    +        from GaussTools.GaussToolsConf import GiGaRunActionExport
    +        export = GiGaRunActionExport("GiGa.GiGaRunActionExport")
    +        giga.RunAction = export
    +        giga.VisManager = "GiGaVisManager/GiGaVis" 
    +
    +        import DetSim
    +        site = 'DayaBay'
    +        DetSim.Configure(physlist=DetSim.physics_list_basic,site=site)
    +
    +
    +
         def configure_optmods(self):
             """ load and configure() "-m" modules here   """
             if self.opts.module == None: return
    @@ -824,6 +876,7 @@
             self.configure_dbi()
             self.configure_job()
             self.configure_geometry()
    +        self.configure_geometry_export()
             self.configure_user()
             self.configure_post_user()
             return
    [blyth@belle7 DybPython]$ 




dae updating
-------------

::

    [blyth@belle7 ~]$ nuwa-
    [blyth@belle7 ~]$ DYB=x nuwa-setup
    [blyth@belle7 cmt]$ cmt show tags | grep geant4_with_dae
    geant4_with_dae (from CMTEXTRATAGS)
    [blyth@belle7 cmt]$ 
    [blyth@belle7 cmt]$ pwd
    /data1/env/local/dybx/NuWa-trunk/dybgaudi/DybRelease/cmt

    [blyth@belle7 cmt]$ cd $(DYB=x nuwa-g4-cmtdir); pwd
    /data1/env/local/dybx/NuWa-trunk/lcgcmt/LCG_Builders/geant4/cmt


::

    cd ~/e/geant4/geometry/DAE
    [blyth@belle7 DAE]$ ./make.sh 
    NB NON-STANDARD DYB x
    Making dependency for file src/G4DAEWriteStructure.cc ...
    ...


After recreating libG4DAE.so it becomes unusable via GaudiPython::

    EventClockSvc.FakeEventTime           INFO Event times generated from 0 with steps of 0
    Generator                             INFO Added gen tool GtTransformTool/onemuonTransformer
    AlgorithmManager                     ERROR Algorithm of type GiGaInputStream is unknown (No factory available).
    AlgorithmManager                     ERROR /data1/env/local/dybx/NuWa-trunk/../external/geant4/4.9.2.p01/i686-slc5-gcc41-dbg/lib/libG4DAE.so: undefined symbol: _ZN15G4DAEWriteAsset10AssetWriteEPN11xercesc_2_810DOMElementE
    AlgorithmManager                     ERROR More information may be available by setting the global jobOpt "ReflexPluginDebugLevel" to 1
    GaudiSequencer                     WARNING Unable to find or create GiGaInputStream
    AlgorithmManager                     ERROR Algorithm of type DsPushKine is unknown (No factory available).
    AlgorithmManager                     ERROR /data1/env/local/dybx/NuWa-trunk/../external/geant4/4.9.2.p01/i686-slc5-gcc41-dbg/lib/libG4DAE.so: undefined symbol: _ZN15G4DAEWriteAsset10AssetWriteEPN11xercesc_2_810DOMElementE
    AlgorithmManager                     ERROR More information may be available by setting the global jobOpt "ReflexPluginDebugLevel" to 1
    GaudiSequencer                     WARNING Unable to find or create DsPushKine
    AlgorithmManager                     ERROR Algorithm of type DsPullEvent is unknown (No factory available).
    AlgorithmManager                     ERROR /data1/env/local/dybx/NuWa-trunk/../external/geant4/4.9.2.p01/i686-slc5-gcc41-dbg/lib/libG4DAE.so: undefined symbol: _ZN15G4DAEWriteAsset10AssetWriteEPN11xercesc_2_810DOMElementE
    AlgorithmManager                     ERROR More information may be available by setting the global jobOpt "ReflexPluginDebugLevel" to 1
    GaudiSequencer                     WARNING Unable to find or create DsPullEvent
    GaudiSequencer                        INFO Member list: 
    EventLoopMgr                         ERROR Unable to initialize Algorithm: GaudiSequencer
    EventLoopMgr                       WARNING Error Initializing base class MinimalEventLoopMgr.
    ServiceManager                       ERROR Unable to initialize Service: EventLoopMgr
    ApplicationMgr                        INFO Application Manager Terminated successfully


BUT Somewhat surprisingly (because the ROOT/Reflex/GaudiPython/.. dict/conf morass is not updated) just doing a clean succeeds::

    cd ~/e/geant4/geometry/DAE
    [blyth@belle7 DAE]$ ./make.sh clean
    NB NON-STANDARD DYB x
    Making dependency for file src/G4DAEWriteStructure.cc ...
    ...



