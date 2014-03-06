Integration of Export Functionality into NuWa
================================================

Commits
--------

* http://dayabay.ihep.ac.cn/tracs/dybsvn/changeset/22635
* http://dayabay.ihep.ac.cn/tracs/dybsvn/changeset/22636


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

See :doc:`geant4/nuwa/dirty_patch` 

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



