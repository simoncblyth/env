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



.. _gdml_build:

Geant4 level manual GDML build
---------------------------------

::

    [blyth@belle7 dyb]$ cd $DYB/external/build/LCG/geant4.9.2.p01/source/persistency/gdml
    [blyth@belle7 gdml]$ make CLHEP_BASE_DIR=$DYB/external/clhep/2.0.4.2/i686-slc5-gcc41-dbg G4SYSTEM=Linux-g++ G4LIB_BUILD_SHARED=1 G4LIB_BUILD_GDML=1 G4LIB_USE_GDML=1 XERCESCROOT=$DYB/external/XercesC/2.8.0/i686-slc5-gcc41-dbg
    Making dependency for file src/G4GDMLWriteStructure.cc ...
    Making dependency for file src/G4GDMLWriteSolids.cc ...
    Making dependency for file src/G4GDMLWriteSetup.cc ...
    ...
    Compiling G4GDMLWriteSetup.cc ...
    Compiling G4GDMLWriteSolids.cc ...
    Compiling G4GDMLWriteStructure.cc ...
    Compiling G4STRead.cc ...
    Creating shared library ../../../lib/Linux-g++/libG4gdml.so ...



Subsequent Geant4Py build misses libG4persistency
---------------------------------------------------

Need to invoke the global target to make that::

    [blyth@belle7 ~]$ cd $DYB/external/build/LCG/geant4.9.2.p01/source/persistency/
    [blyth@belle7 persistency]$ vi ../../config/globlib.gmk
    [blyth@belle7 persistency]$ make CLHEP_BASE_DIR=$DYB/external/clhep/2.0.4.2/i686-slc5-gcc41-dbg G4SYSTEM=Linux-g++ G4LIB_BUILD_SHARED=1 G4LIB_BUILD_GDML=1 G4LIB_USE_GDML=1 XERCESCROOT=$DYB/external/XercesC/2.8.0/i686-slc5-gcc41-dbg global
    Nothing to be done for libG4persistency in mctruth/.
    Nothing to be done for libG4persistency in ascii/.
    Nothing to be done for libG4persistency in gdml/.
    Creating global shared library ../../lib/Linux-g++/libG4persistency.so ...
    [blyth@belle7 persistency]$ 


.. _gdml_install:

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


Manually install G4DAE includes into NuWa/G4 location
--------------------------------------------------------

* perhaps a G4 dybinst patch is the eay to go for G4DAE, rather than this manual juggling 

::

    blyth@belle7 cmt]$ t dae-install-inc
    dae-install-inc is a function
    dae-install-inc () 
    { 
        nuwa-;
        local blib=$(env-home)/geant4/geometry/DAE/include;
        local ilib=$(nuwa-g4-idir)/include;
        ls --color=tty -l $blib;
        local cmd="cp $blib/G4DAE*.{hh,icc} $ilib/";
        echo "$cmd";
        eval $cmd;
        ls --color=tty -l $ilib/G4DAE*
    }


Build modified GaussTools 
--------------------------

With G4GDML and G4DAE switched on::

    [blyth@belle7 cmt]$ svn diff
    Index: requirements
    ===================================================================
    --- requirements        (revision 21750)
    +++ requirements        (working copy)
    @@ -31,6 +31,9 @@
     apply_pattern     component_library library=GaussTools
     apply_pattern     linker_library    library=GaussToolsLib
     
    +# SCB : enable GDML and DAE export by GiGaRunActionGDML
    +macro_append GaussTools_cppflags " -DHAVE_G4GDML=1 -DHAVE_G4DAE=1 "
    +macro_append GaussTools_linkopts " -lG4DAE "
    +

::

    [blyth@belle7 cmt]$ pwd
    /data1/env/local/dyb/NuWa-trunk/lhcb/Sim/GaussTools/cmt
    [blyth@belle7 cmt]$ fenv
    [blyth@belle7 cmt]$ cmt config
    Removing all previous make fragments from i686-slc5-gcc41-dbg
    Creating setup scripts.
    Creating cleanup scripts.
    cmt directory already installed
    src directory already installed
    doc directory already installed
    GaussTools directory already installed
    [blyth@belle7 cmt]$ . setup.sh
    [blyth@belle7 cmt]$ pwd
    /data1/env/local/dyb/NuWa-trunk/lhcb/Sim/GaussTools/cmt
    [blyth@belle7 cmt]$ export VERBOSE=1
    [blyth@belle7 cmt]$ cmt make






.. _gdml_export:

Perform Export creating 3.2M file
--------------------------------------

::

    # --- GDML geometry export ---------------------------------
    #   
    from GaussTools.GaussToolsConf import GiGaRunActionGDML
    grag = GiGaRunActionGDML("GiGa.GiGaRunActionGDML")
    giga = GiGa()
    giga.RunAction = grag    


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


Perform the export::

    [blyth@belle7 gdml]$ cd ~/e/geant4/geometry/gdml
    [blyth@belle7 gdml]$ fenv
    [blyth@belle7 gdml]$ ./export.sh


NuWa integration GDML export functionality
--------------------------------------------

``GiGaRunActionGDML`` is floating in working copy on N, as expect it will
not compile/link with standard NuWa with GDML switched OFF.::

    [blyth@belle7 Components]$ st 
    A  +    GiGaRunActionGDML.cpp
    A  +    GiGaRunActionGDML.h
    [blyth@belle7 Components]$ pwd
    /data1/env/local/dyb/NuWa-trunk/lhcb/Sim/GaussTools/src/Components

Rejig the code so it will compile and do nothing when ``HAVE_G4DAE`` and ``HAVE_G4GDML`` are not set. 

The headers and lib are present on C (I dont recall tampering there, but it seems I must have)::

    [blyth@cms01 include]$ l G4GDML*
    ...
    -rw-r--r--  1 blyth blyth 2300 Oct  1 18:03 G4GDMLWriteSetup.hh
    -rw-r--r--  1 blyth blyth 4894 Oct  1 18:03 G4GDMLWriteSolids.hh
    -rw-r--r--  1 blyth blyth 3306 Oct  1 18:03 G4GDMLWriteStructure.hh
    [blyth@cms01 include]$ pwd
    /data/env/local/dyb/trunk/external/geant4/4.9.2.p01/i686-slc4-gcc34-dbg/include

    [blyth@cms01 lib]$ l libG4gdml.so 
    -rwxrwxr-x  1 blyth blyth 592988 Oct  1 18:03 libG4gdml.so

    [blyth@cms01 lib]$ pwd
    /data/env/local/dyb/trunk/external/geant4/4.9.2.p01/i686-slc4-gcc34-dbg/lib

How to preprocessor protect entire classes with NuWa/cmt ? 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It would be good to avoid branching just for a few extra classes, if can 
avoid interference.

Example of private test code NuWa-trunk/dybgaudi/RootIO/RootIOSvc/cmt/requirements::

     28 # Below is for test code
     29 
     30 private
     31 
     32 pattern  public_include include_dirs ${<package>_cmtpath}/include
     33 pattern private_include include_dirs ../include
     34 pattern test_application \
     35         public ; \
     36         apply_pattern executable_path ; \
     37         private ; \
     38         apply_pattern public_include ; \
     39         application <name> -group=test <files>
     40 
     41 
     42 
     43 use CLHEP v* LCG_Interfaces         # in lcgcmt
     44 use CLHEPRflx v* Dictionaries           # in relax
     45 
     46 macro_append ROOT_linkopts " -lCLHEPRflx -lCintex "
     47 test_application name=RootIOTest files=../test-bin/RootIOTest.cc
     48 macro_append RootIOTestlinkopts " -L$(RootIODir) -lRootIO "
     49 


Example pre-processor setting NuWa-trunk/dybgaudi/DaqFormat/RawRecordPool/cmt/requirements::

      1 package RawRecordPool
      2 
      3 use DybPolicy
      4 
      5 library RawRecordPool  *.cc
      6 
      7 macro_append RawRecordPool_cppflags " -DRPC_ERROR_DEBUG=10 "
      8 
      9 apply_pattern install_more_includes more=RawRecordPool




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


Annoying physvol name truncation + uniqing
---------------------------------------------

#. physvol names are trucated to 99 chars
#. names are sometimes but not always uniqued by appending pointer address

::

    30017       <physvol name="/dd/Geometry/Pool/lvNearPoolOWS#pvNearUnistruts#pvNearHalfUnistruts:2#pvNearQuadCornerUnistrus:1#pv">
    30018         <volumeref ref="/dd/Geometry/PoolDetails/lvCornerParRib20xbc9ee78"/>
    30019         <position name="/dd/Geometry/Pool/lvNearPoolOWS#pvNearUnistruts#pvNearHalfUnistruts:2#pvNearQuadCornerUnistrus:1#pv" unit="mm" x="-5743.97485196094" y="-2743.97485196094" z="1944"/>
    30020         <rotation name="/dd/Geometry/Pool/lvNearPoolOWS#pvNearUnistruts#pvNearHalfUnistruts:2#pvNearQuadCornerUnistrus:1#pv" unit="deg" x="0" y="0" z="-135"/>
    30021       </physvol>


Yep traversing the gdml and dumping physvol names, confirms the truncation::

    71 /dd/Geometry/AdDetails/lvCtrLsoOflInOil#pvCtrGdsOflTfbInLsoOfl0xbe03178
    68 /dd/Geometry/AdDetails/lvOcrGdsTfbInLsoOfl#pvOcrGdsInLsoOfl0xbadf408
    66 /dd/Geometry/AdDetails/lvOcrGdsLsoOfl#pvOcrGdsTfbInLsoOfl0xbb77f40
    82 /dd/Geometry/CalibrationSources/lvWallLedSourceAssy#pvWallLedDiffuserBall0xc065178
    80 /dd/Geometry/CalibrationSources/lvWallLedSourceAssy#pvWallLedAcrylicRod0xbd92aa0
    36 /dd/Geometry/AD/lvOIL#pvOAV0xbf66370
    99 /dd/Geometry/AD/lvOIL#pvAdPmtArray#pvAdPmtArrayRotated#pvAdPmtRingInCyl:1#pvAdPmtInRing:1#pvAdPmtUn
    99 /dd/Geometry/AD/lvOIL#pvAdPmtArray#pvAdPmtArrayRotated#pvAdPmtRingInCyl:1#pvAdPmtInRing:1#pvAdPmtUn
    99 /dd/Geometry/AD/lvOIL#pvAdPmtArray#pvAdPmtArrayRotated#pvAdPmtRingInCyl:1#pvAdPmtInRing:2#pvAdPmtUn
    99 /dd/Geometry/AD/lvOIL#pvAdPmtArray#pvAdPmtArrayRotated#pvAdPmtRingInCyl:1#pvAdPmtInRing:2#pvAdPmtUn
    99 /dd/Geometry/AD/lvOIL#pvAdPmtArray#pvAdPmtArrayRotated#pvAdPmtRingInCyl:1#pvAdPmtInRing:3#pvAdPmtUn
    99 /dd/Geometry/AD/lvOIL#pvAdPmtArray#pvAdPmtArrayRotated#pvAdPmtRingInCyl:1#pvAdPmtInRing:3#pvAdPmtUn


Compare against the shape shapes from the VRML2FILE export::

    sqlite> select name,length(name) from shape where name like '/dd/Geometry/Pool/lvNearPoolOWS#pvNearUnistruts#pvNearHalfUnistruts:2#pvNearQuadCornerUnistrus:1#pv%' ;
    name                                                                                                                                                                                                      length(name)
    ---------------------------------------------------------------------------------------------                                                                                                             ------------
    /dd/Geometry/Pool/lvNearPoolOWS#pvNearUnistruts#pvNearHalfUnistruts:2#pvNearQuadCornerUnistrus:1#pvCornerUnistrut1#pvBotCornerLinerParRib1.2                                                              140         
    /dd/Geometry/Pool/lvNearPoolOWS#pvNearUnistruts#pvNearHalfUnistruts:2#pvNearQuadCornerUnistrus:1#pvCornerUnistrut1#pvBotCornerLinerParRib2.2                                                              140         
    /dd/Geometry/Pool/lvNearPoolOWS#pvNearUnistruts#pvNearHalfUnistruts:2#pvNearQuadCornerUnistrus:1#pvCornerUnistrut1#pvBotCornerCurtainParRib1.2                                                            142         
    /dd/Geometry/Pool/lvNearPoolOWS#pvNearUnistruts#pvNearHalfUnistruts:2#pvNearQuadCornerUnistrus:1#pvCornerUnistrut1#pvBotCornerCurtainParRib2.2                                                            142         
    /dd/Geometry/Pool/lvNearPoolOWS#pvNearUnistruts#pvNearHalfUnistruts:2#pvNearQuadCornerUnistrus:1#pvCornerUnistrut1#pvBotCornerLinerVerRib1.2                                                              140         
    /dd/Geometry/Pool/lvNearPoolOWS#pvNearUnistruts#pvNearHalfUnistruts:2#pvNearQuadCornerUnistrus:1#pvCornerUnistrut1#pvBotCornerLinerVerRib2.2                                                              140         
    /dd/Geometry/Pool/lvNearPoolOWS#pvNearUnistruts#pvNearHalfUnistruts:2#pvNearQuadCornerUnistrus:1#pvCornerUnistrut1#pvBotCornerCurtainVerRib1#pvBotVertiRibUnit.2                                          160         
    /dd/Geometry/Pool/lvNearPoolOWS#pvNearUnistruts#pvNearHalfUnistruts:2#pvNearQuadCornerUnistrus:1#pvCornerUnistrut1#pvBotCornerCurtainVerRib2#pvBotVertiRibUnit.2                                          160         
    /dd/Geometry/Pool/lvNearPoolOWS#pvNearUnistruts#pvNearHalfUnistruts:2#pvNearQuadCornerUnistrus:1#pvCornerUnistrut1#pvBotCornerLinerVerRib0.2                                                              140         


Hunt the truncation::


    [blyth@cms01 src]$ pwd
    /data/env/local/dyb/trunk/external/build/LCG/geant4.9.2.p01/source/persistency/gdml/src

    [blyth@cms01 src]$ grep 99 *.*
    G4GDMLWrite.cc:   xercesc::XMLString::transcode(name,tempStr,99);
    G4GDMLWrite.cc:   xercesc::XMLString::transcode(value,tempStr,99);
    G4GDMLWrite.cc:   xercesc::XMLString::transcode(name,tempStr,99);
    G4GDMLWrite.cc:   xercesc::XMLString::transcode(str,tempStr,99);
    G4GDMLWrite.cc:   xercesc::XMLString::transcode(name,tempStr,99);
    G4GDMLWrite.cc:   xercesc::XMLString::transcode("LS", tempStr, 99);
    G4GDMLWrite.cc:   xercesc::XMLString::transcode("Range", tempStr, 99);
    G4GDMLWrite.cc:   xercesc::XMLString::transcode("gdml", tempStr, 99);
    [blyth@cms01 src]$ 



Rebuild GDML with simple fix
-----------------------------

::

    [blyth@belle7 gdml]$ cd $DYB/external/build/LCG/geant4.9.2.p01/source/persistency/gdml/src
    [blyth@belle7 src]$  vi ../include/G4GDMLWrite.hh


::

    099  private:
    100 
    101    static G4bool addPointerToName;
    102    xercesc::DOMDocument* doc;
    103    //XMLCh tempStr[100];
           const static int tempStrSize = 256;
           XMLCh tempStr[tempStrSize];
           
    104 
    105 };


::

    80    xercesc::XMLString::transcode(name,tempStr,99);
    :.,$s,99,tempStrSize-1,gc


Repeat the  above steps:

* :ref:`gdml_build`
* :ref:`gdml_install`
* :ref:`gdml_export`


Confirm the fix
------------------

::

    [blyth@belle7 gdml]$ du -hs g4_00.gdml
    4.0M    g4_00.gdml
    [blyth@belle7 gdml]$ mv  g4_00.gdml  $LOCAL_BASE/env/geant4/geometry/gdml/g4_01.gdml


Address uniqing causes too many diffs.

::

    simon:gdml blyth$ scp N:/data1/env/local/env/geant4/geometry/gdml/g4_01.gdml  $LOCAL_BASE/env/geant4/geometry/gdml/



Consolidated DAE and GDML export
---------------------------------

::

    [blyth@belle7 gdml]$ pwd
    /home/blyth/e/geant4/geometry/gdml
    [blyth@belle7 gdml]$ cat export.sh
    #!/bin/sh
    cd $ENV_HOME/geant4/geometry/gdml
    nuwa.py -G $XMLDETDESCROOT/DDDB/dayabay.xml -n1 -m export
    [blyth@belle7 gdml]$ ./export.sh 

::

    ...
    GiGaRunActionGDML::BeginOfRunAction writing GDML to g4_00.gdml
    G4GDML: Writing 'g4_00.gdml'...
    G4GDML: Writing definitions...
    G4GDML: Writing materials...
    G4GDML: Writing solids...
    G4GDML: Writing structure...
    G4GDML: Writing setup...
    G4GDML: Writing 'g4_00.gdml' done !
    GiGaRunActionGDML::BeginOfRunAction writing COLLADA to g4_00.dae
    G4DAE: Writing 'g4_00.dae'...
    G4DAE: Writing asset metadata...
    G4DAE: Writing library_effects...
    G4DAE: Writing library_geometries...
    G4DAE: Writing library_materials...
    G4DAE: Writing structure/library_nodes...
    G4DAE: Writing library_visual_scenes...
    G4DAEWriteStructure::PhysvolWrite /dd/Geometry/Sites/lvNearHallTop#pvNearTopCover0xbd581c0 1000
    G4DAEWriteStructure::PhysvolWrite /dd/Geometry/RPC/lvRPCGasgap14#pvStrip14Array#pvStrip14ArrayOne:1#pvStrip14Unit0xc2244b0 1
    G4DAEWriteStructure::PhysvolWrite /dd/Geometry/RPC/lvRPCGasgap14#pvStrip14Array#pvStrip14ArrayOne:2#pvStrip14Unit0xc225438 2
    ...
    G4DAEWriteStructure::PhysvolWrite /dd/Geometry/RPCSupport/lvNearHbeamBigUnit#pvNearRightSpanHbeam10xbe18010 1001
    G4DAEWriteStructure::PhysvolWrite /dd/Geometry/RPCSupport/lvNearHbeamBigUnit#pvNearLeftSpanHbeam20xbe180b0 1002
    G4DAEWriteStructure::PhysvolWrite /dd/Geometry/RPCSupport/lvNearHbeamBigUnit#pvNearRightSpanHbeam20xbe18150 1003
    BooleanProcessor::triangulateContour : could not generate a triangle (infinite loop)
    BooleanProcessor: boolean operation failed
    G4DAEWriteStructure::PhysvolWrite /dd/Geometry/RPCSupport/lvNearHbeamBigUnit#pvNearUpSideBigHbeam0xbe18188 1004
    G4DAEWriteStructure::PhysvolWrite /dd/Geometry/RPCSupport/lvNearHbeamBigUnit#pvNearDownSideBigHbeam0xbe18590 1005
    G4DAEWriteStructure::PhysvolWrite /dd/Geometry/RPCSupport/lvNearHbeamBigUnit#pvNearThwartLongAILeftUpY10xbe18cb0 1006
    ...


::

    [blyth@belle7 gdml]$ du -h g4_00.*
    5.0M    g4_00.dae
    4.0M    g4_00.gdml




