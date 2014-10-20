
G4DAEChroma
=============

Objective
------------

Pull out everything Chroma related and reusable 
from StackAction and SensitiveDetector
for flexible reusability from different Geant4 contexts

Dependencies
------------

* giga/gaudi/gauss NOT ALLOWED 
* sticking to plain Geant4, ZMQ, ZMQRoot,... for generality 

How to organize code
-----------------------

Everything under single G4DAEChroma umbrella manager singleton
using methods and constituent worker classes for each aspect: 

* geometry export 
* geometry gdml loading
* trojan SD registration
* extra hit adding 
* photon collection 

Primary concern of organization is:

* **ease of testing from MockNuWa**

Leave usage of StackAction etc at level of examples


Initialize in RunAction?
--------------------------

::

   // 2nd parameter target must match the name of an existing SD 

Normally `AddNewDetector` is done at G4 ConstructDetector 
initialization stage but seems no GiGa hooks back then. 
Try in RunAction, but with a check to make sure not already there.
Makes sense to add this to the `GiGaRunActionExport` code that does the G4DAE export..

* http://dayabay.ihep.ac.cn/tracs/dybsvn/browser/lhcb/trunk/Sim/GaussTools/src/Components
* http://dayabay.ihep.ac.cn/tracs/dybsvn/browser/lhcb/trunk/Sim/GaussTools/src/Components/GiGaRunActionExport.cpp

As operating from the real G4 geometry tree (not the GDML one), 
can collect SD names by logical volume inspection during the traverse. 
Might as well include SD names in the COLLADA export metadata.


Looking for hooks
~~~~~~~~~~~~~~~~~

::

    [blyth@cms01 lhcb]$ find . -name '*Action.h'
    ./InstallArea/include/GiGa/IGiGaEventAction.h
    ./InstallArea/include/GiGa/IGiGaStepAction.h
    ./InstallArea/include/GiGa/IGiGaStackAction.h
    ./InstallArea/include/GiGa/IGiGaTrackAction.h
    ./InstallArea/include/GiGa/IGiGaRunAction.h
    ./Sim/GiGa/src/Lib/IIDIGiGaRunAction.h
    ./Sim/GiGa/src/Lib/IIDIGiGaTrackAction.h
    ./Sim/GiGa/src/Lib/IIDIGiGaEventAction.h
    ./Sim/GiGa/src/Lib/IIDIGiGaStepAction.h
    ./Sim/GiGa/src/Lib/IIDIGiGaStackAction.h
    ./Sim/GiGa/GiGa/IGiGaEventAction.h
    ./Sim/GiGa/GiGa/IGiGaStepAction.h
    ./Sim/GiGa/GiGa/IGiGaStackAction.h
    ./Sim/GiGa/GiGa/IGiGaTrackAction.h
    ./Sim/GiGa/GiGa/IGiGaRunAction.h
    ./Sim/GaussTools/src/Components/CommandTrackAction.h
    ./Sim/GaussTools/src/Components/TrCutsRunAction.h
    ./Sim/GaussTools/src/Components/GaussStepAction.h
    ./Sim/GaussTools/src/Components/GaussPostTrackAction.h
    ./Sim/GaussTools/src/Components/GaussPreTrackAction.h
    ./Sim/GaussTools/src/Components/CutsStepAction.h
    [blyth@cms01 lhcb]$ 


`env/geant4/geometry/export/export_all.py`::

     70     from GaussTools.GaussToolsConf import GiGaRunActionExport, GiGaRunActionCommand, GiGaRunActionSequence
     71     export = GiGaRunActionExport("GiGa.GiGaRunActionExport")
     72 
     73     #   NOT WORKING :  RunSeq fails to do the vis : only the GDML+DAE gets exported
     74     #   so do at C++ level 
     75     #
     76     #wrl  = GiGaRunActionCommand("GiGa.GiGaRunActionCommand")
     77     #wrl.BeginOfRunCommands = [ 
     78     #         "/vis/open VRML2FILE",
     79     #         "/vis/viewer/set/culling global false",
     80     #         "/vis/viewer/set/culling coveredDaughters false",
     81     #         "/vis/drawVolume",
     82     #         "/vis/viewer/flush"
     83     #] 
     84     #runseq = GiGaRunActionSequence("GiGa.GiGaRunActionSequence")
     85     #giga.addTool( runseq , name="RunSeq" )
     86     #giga.RunSeq.Members += ["GiGaRunActionCommand"]
     87     #giga.RunSeq.Members += ["GiGaRunActionGDML"]
     88     #giga.RunAction = "GiGaRunActionSequence/RunSeq"     
     89     # why so many ways to address things ? Duplication is evil  
     90 
     91     giga.RunAction = export



Issues
--------

Development Cycle too slow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create test application for machinery test 
(enable to rapidly work on the marshalling) 

* reads Dyb geometry into G4 from exported GDML
* reads some initial photon positions from a .root file
* invokes this photon collection and propagation 
* dumps the hits returned

**Using MockNuWa with NuWa DataModel subset for fast cycle**


GPU Hit handling : SensDet
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* how to register DsChromaPmtSensDet instead of (or in addition to) DsPmtSensDet
  or some how get access to DsPmtSensDet

  * class name "DsPmtSensDet" is mentioned in DetDesc 
    logvol sensdet attribute, somehow DetDesc/GiGa 
    hands that over to Geant4 : need to swizzle OR add ? 

  * old approach duplicated bits of "DsPmtSensDet" for adding 
    hits into the StackAction : that was too messy then, but perhaps
    clean enough now have pulled out Chroma parts into G4DAEChroma 

  * but needs access to private methods from DsPmtSensDet, so 
    maybe a no-no anyhow : especially as need very little
    functionality 

**Using TrojanSD approach registered in the RunActionExport**


Accessing SD
~~~~~~~~~~~~~~~~

* how to get access to DsPmtSensDet in order to add hits

  * provide singleton accessor for cheat access to globally 
    shared instance ? 
    Approach has MT complications : but no need to worry about that yet

  * gaudi has a way of accessing the instance, do it externally (where?)
    and pass it in 


**Doing it via a Trojan parasitic G4VSensitiveDetector which 
caches the hit collections of the real SD**::

   // adding extra hits needs access to the tsd
   TrojanSensDet* TSD = (TrojanSensDet*)G4SDManager::GetSDMpointer()->FindSensitiveDetector("Trojan_DsPmtSensDet", true); 



Detector Specific Code
~~~~~~~~~~~~~~~~~~~~~~~

* how to handle hits interfacing to detector specific code

* arrange det specifics together and use preprocessor macros



No point duplicating hit
--------------------------

::

    struct Hit {
        // global
        G4ThreeVector gpos ;
        G4ThreeVector gdir ;
        G4ThreeVector gpol ;

       // local : maybe just keep local, inplace transform ?
        G4ThreeVector lpos ;
        G4ThreeVector ldir ;
        G4ThreeVector lpol ;

        float t ;
        float wavelength ;
        int   hitindex ; 
        int   pmtid ;
        int   volumeindex ;

        void LocalTransform(G4AffineTransform& trans)
        { 
            lpos = trans.TransformPoint(gpos);
            lpol = trans.TransformAxis(gpol);
            lpol = lpol.unit();
            ldir = trans.TransformAxis(gdir);
            ldir = ldir.unit();
        }
        void Print(){
              G4cout 
                     << " hitindex " << hitindex 
                     << " volumeindex " << volumeindex 
                     << " pmtid "       << pmtid 
                     << " t "     << t 
                     << " wavelength " << wavelength 
                     << " gpos "  << gpos 
                     << " gdir "  << gdir 
                     << " gpol "  << gpol 
                     << G4endl ; 
         }
    }; 





