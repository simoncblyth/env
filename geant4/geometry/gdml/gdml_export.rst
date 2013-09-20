Export GDML 
=============

.. contents:: :local:

GDML is switched off in NuWa
-------------------------------

libG4gdml.so is built with geant4 if the switch is ON NuWa-trunk/lcgcmt/LCG_Builders/geant4/cmt/requirements::

     90 set G4LIB_BUILD_GDML "1" \
     91     dayabay ""
     92 
     93 set G4LIB_USE_GDML "1" \
     94     dayabay ""

Geant4 level manual GDML build
---------------------------------

::

    [blyth@belle7 dyb]$ cd external/build/LCG/geant4.9.2.p01/source/persistency/gdml
    [blyth@belle7 gdml]$ make CLHEP_BASE_DIR=/data1/env/local/dyb/external/clhep/2.0.4.2/i686-slc5-gcc41-dbg G4SYSTEM=Linux-g++ G4LIB_BUILD_SHARED=1 G4LIB_BUILD_GDML=1 G4LIB_USE_GDML=1 XERCESCROOT=/data1/env/local/dyb/external/XercesC/2.8.0/i686-slc5-gcc41-dbg
    Making dependency for file src/G4GDMLWriteStructure.cc ...
    Making dependency for file src/G4GDMLWriteSolids.cc ...
    Making dependency for file src/G4GDMLWriteSetup.cc ...
    ...
    Compiling G4GDMLWriteSetup.cc ...
    Compiling G4GDMLWriteSolids.cc ...
    Compiling G4GDMLWriteStructure.cc ...
    Compiling G4STRead.cc ...
    Creating shared library ../../../lib/Linux-g++/libG4gdml.so ...

GDML Manual Install lib and includes
-------------------------------------

::

    [blyth@belle7 gdml]$  cp ../../../lib/Linux-g++/libG4gdml.so $DYB/NuWa-trunk/../external/geant4/4.9.2.p01/i686-slc5-gcc41-dbg/lib/libG4gdml.so

    [blyth@belle7 gdml]$ l $DYB/external/geant4/4.9.2.p01/i686-slc5-gcc41-dbg/include/G4ST*
    -rw-r--r-- 1 blyth blyth 2249 Mar 16  2009 /data1/env/local/dyb/external/geant4/4.9.2.p01/i686-slc5-gcc41-dbg/include/G4STEPEntity.hh
    [blyth@belle7 gdml]$ l $DYB/external/geant4/4.9.2.p01/i686-slc5-gcc41-dbg/include/G4GDML*
    ls: /data1/env/local/dyb/external/geant4/4.9.2.p01/i686-slc5-gcc41-dbg/include/G4GDML*: No such file or directory
    [blyth@belle7 gdml]$ cp include/* $DYB/external/geant4/4.9.2.p01/i686-slc5-gcc41-dbg/include/


GDML via GiGa
--------------

* :google:`geant4 giga gdml`

    * http://svn.cern.ch/guest/lhcb/Gauss/trunk/Sim/LbGDML/options/GDMLWriter.opts
    * http://svn.cern.ch/guest/lhcb/packages/trunk/Sim/GDMLG4Writer/src/GDMLRunAction.cpp

Find something relevant. API has changed, but principal is the same. Mostly GiGa glue code.

::

    [blyth@belle7 NuWa-trunk]$ find . -name '*.cpp' -exec grep -l RunAction {} \;
    ./lhcb/Sim/GaussTools/src/Components/GiGaRunActionCommand.cpp
    ./lhcb/Sim/GaussTools/src/Components/TrCutsRunAction.cpp
    ./lhcb/Sim/GaussTools/src/Components/GaussTools_load.cpp
    ./lhcb/Sim/GaussTools/src/Components/GiGaRunActionSequence.cpp
    ./lhcb/Sim/GiGa/src/Lib/GiGaInterfaces.cpp
    ./lhcb/Sim/GiGa/src/Lib/GiGaRunActionBase.cpp
    ./lhcb/Sim/GiGa/src/component/GiGa.cpp
    ./lhcb/Sim/GiGa/src/component/GiGaRunManagerInterface.cpp
    ./lhcb/Sim/GiGa/src/component/GiGaIGiGaSetUpSvc.cpp
    ./lhcb/Sim/GiGa/src/component/GiGaRunManager.cpp
    [blyth@belle7 NuWa-trunk]$ 


Identify something similar `lhcb/Sim/GaussTools/src/Components/GiGaRunActionCommand.cpp` 
to base `GiGaRunActionGDML` upon and piggyback the CMT controlled build, from::

    [blyth@belle7 cmt]$ pwd
    /data1/env/local/dyb/NuWa-trunk/lhcb/Sim/GaussTools/cmt


Invoke from python creating 3.2M file
--------------------------------------

::

    # --- GDML geometry export ---------------------------------
    #   
    from GaussTools.GaussToolsConf import GiGaRunActionGDML
    grag = GiGaRunActionGDML("GiGa.GiGaRunActionGDML")
    giga = GiGa()
    giga.RunAction = grag    


.. literalinclude:: export.sh

.. literalinclude:: export.py


::

    GiGaRunActionGDML::BeginOfRunAction writing to 
    G4GDML: Writing 'g4_00.gdml'...
    G4GDML: Writing definitions...
    G4GDML: Writing materials...
    G4GDML: Writing solids...
    G4GDML: Writing structure...
    G4GDML: Writing setup...
    G4GDML: Writing 'g4_00.gdml' done !
    Start Run processing.




Cursory Look
-------------


::

    [blyth@belle7 gdml]$ du -h g4_00.gdml 
    3.2M    g4_00.gdml

    [blyth@belle7 gdml]$ wc -l g4_00.gdml 
    30946 g4_00.gdml

    [blyth@belle7 gdml]$ head -15 g4_00.gdml 
    <?xml version="1.0" encoding="UTF-8" standalone="no" ?>
    <gdml xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://service-spi.web.cern.ch/service-spi/app/releases/GDML/schema/gdml.xsd">

      <define/>

      <materials>
        <element Z="6" name="/dd/Materials/Carbon0xbe238a8">
          <atom unit="g/mole" value="12.0109936803044"/>
        </element>
        <element Z="1" name="/dd/Materials/Hydrogen0xbe22480">
          <atom unit="g/mole" value="1.00793946966331"/>
        </element>
        <material name="/dd/Materials/PPE0xbb80090" state="solid">
          <P unit="pascal" value="101324.946686941"/>
          <D unit="g/cm3" value="0.919999515933733"/>

    [blyth@belle7 gdml]$ tail  -15 g4_00.gdml 
          <materialref ref="/dd/Materials/Vacuum0xbe4e7d8"/>
          <solidref ref="WorldBox0xc9818a0"/>
          <physvol name="/dd/Structure/Sites/db-rock0xc982aa8">
            <volumeref ref="/dd/Geometry/Sites/lvNearSiteRock0xbb7d528"/>
            <position name="/dd/Structure/Sites/db-rock0xc982aa8_pos" unit="mm" x="-16519.9999999999" y="-802110" z="-2110"/>
            <rotation name="/dd/Structure/Sites/db-rock0xc982aa8_rot" unit="deg" x="0" y="0" z="-122.9"/>
          </physvol>
        </volume>
      </structure>

      <setup name="Default" version="1.0">
        <world ref="World0xc982758"/>
      </setup>

    </gdml>



Obnoxious uniqing 
~~~~~~~~~~~~~~~~~~~~

::

     3877     <volume name="/dd/Geometry/AD/lvLSO0xbe14900">
     3878       <materialref ref="/dd/Materials/LiquidScintillator0xbf257f8"/>
     3879       <solidref ref="lso0xbba9ff8"/>
     3880       <physvol name="/dd/Geometry/AD/lvLSO#pvIAV0xbb2e4a8">
     3881         <volumeref ref="/dd/Geometry/AD/lvIAV0xbe18188"/>
     3882         <position name="/dd/Geometry/AD/lvLSO#pvIAV0xbb2e4a8_pos" unit="mm" x="0" y="0" z="2.5"/>
     3883       </physvol>


$DYB/external/build/LCG/geant4.9.2.p01/examples/extended/persistency/gdml/G02/src/DetectorConstruction.cc::

    156     // OPTION: SETTING ADDITION OF POINTER TO NAME TO FALSE
    157     //
    158     // By default, written names in GDML consist of the given name with
    159     // appended the pointer reference to it, in order to make it unique.
    160     // Naming policy can be changed by using the following method, or
    161     // calling Write with additional Boolean argument to "false".
    162     // NOTE: you have to be sure not to have duplication of names in your
    163     //       Geometry Setup.
    164     // 
    165     // parser.SetAddPointerToName(false);
    166     //
    167     // or
    168     //
    169     // parser.Write(fWriteFile, fWorldPhysVol, false);
    170 



