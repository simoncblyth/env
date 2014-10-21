
G4DAEChroma
=============

Objective
------------

Pull out everything Chroma related and reusable 
from StackAction and SensitiveDetector
for flexible reusability from different Geant4 contexts

Dependencies
------------

* ROOT, CLHEP, Geant4, ZMQ, ZMQRoot
* **NOT** detsim/giga/gaudi/gauss

  * G4DAEChroma intended to be used from minimal 
    shim G4 Actions hooked up in DetSim/GiGa 
    in as few lines of code as possible

  * this facilities limited dependency development/testing 
    providing faster dev cycle : see `mocknuwa-` `datamodel-`


Usage Stages
--------------

BeginOfRunAction
~~~~~~~~~~~~~~~~~~

* `G4DAEChroma::G4DAEChroma` ctor singleton and configure constituents

  * Geometry : traverse volume tree creating tranform cache 
  * Transport : prepare ZMQ sockets  
  * SensDet : use Trojan to steal DsPmtSensDet hit collections and AddNewDetector 

* geometry is exported to DAE in BeginOfRunAction also, 
  but only need to do that once


StackAction::ClassifyNewTrack OR customized Processes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* `G4DAEChroma::CollectPhoton`  and avoid further 
   G4 processing via kill or not adding as secondary 


StackAction::NewStage  
~~~~~~~~~~~~~~~~~~~~~~~

* perhaps could be Event 

* `G4DAEChroma::Propagate`  which proceeds to add the hits 


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


MockNuWa code development
---------------------------

See

* mocknuwa-
* datamodel-
* gdc-  G4DAEChroma
* gdct- G4DAEChromaTest

Real NuWa hookup for machinery test
--------------------------------------

::

    [blyth@belle7 dybgaudi]$ svn ci -m "minor: add G4DAEChroma package and hookup to DetSimChroma StackAction and RunAction "
    Sending        Simulation/DetSimChroma/src/DetSimChroma_entries.cc
    Adding         Simulation/DetSimChroma/src/DsChromaRunAction.cc
    Adding         Simulation/DetSimChroma/src/DsChromaRunAction.h
    Sending        Simulation/DetSimChroma/src/DsChromaStackAction.cc
    Adding         Utilities/G4DAEChroma/G4DAEChroma/G4DAEChroma.hh
    Adding         Utilities/G4DAEChroma/G4DAEChroma/G4DAEGeometry.hh
    Adding         Utilities/G4DAEChroma/G4DAEChroma/G4DAESensDet.hh
    Adding         Utilities/G4DAEChroma/G4DAEChroma/G4DAETransport.hh
    Adding         Utilities/G4DAEChroma/G4DAEChroma/G4DAETrojanSensDet.hh
    Sending        Utilities/G4DAEChroma/cmt/requirements
    Sending        Utilities/G4DAEChroma/src/G4DAEChroma.cc
    Adding         Utilities/G4DAEChroma/src/G4DAEGeometry.cc
    Adding         Utilities/G4DAEChroma/src/G4DAESensDet.cc
    Adding         Utilities/G4DAEChroma/src/G4DAETransport.cc
    Adding         Utilities/G4DAEChroma/src/G4DAETrojanSensDet.cc
    Transmitting file data ...............
    Committed revision 23458.
    [blyth@belle7 dybgaudi]$ date
    Tue Oct 21 20:57:27 CST 2014








Integrate with real NuWa via shims:

* `DsChromaRunAction` 
* `DsChromaStackAction`

that all depend on G4DAEChroma from Utilities.

Keep all functionality in G4DAEChroma, only thing admissable
to do in the shim is configuration.


csa : ChromaStackAction
~~~~~~~~~~~~~~~~~~~~~~~~~

Hmm this is sourced from people area SVN, move to env.

/data1/env/local/env/muon_simulation/optical_photon_weighting/OPW/fmcpmuon.py::

    321     def configure_chromastackaction(self):
    322         log.info("configure_chromastackaction")
    323         import DetSimChroma
    324         from DetSimChroma.DetSimChromaConf import DsChromaStackAction
    325         saction = DsChromaStackAction("GiGa.DsChromaStackAction")
    326         saction.PhotonCut = True      # kill OP after collection
    327         saction.ModuloPhoton = 1000   # scale down collection
    328         return saction

export- 
~~~~~~~~~

Handled by adding RunAction sourced from GaussTools, but cannot make GaussTools 
depend on G4DAEChroma

`env/geant4/geometry/export/export_all.py`::

     69     # --- WRL + GDML + DAE geometry export ---------------------------------
     70     from GaussTools.GaussToolsConf import GiGaRunActionExport, GiGaRunActionCommand, GiGaRunActionSequence
     71     export = GiGaRunActionExport("GiGa.GiGaRunActionExport")
     ..
     91     giga.RunAction = export



GiGaRunActionBase
~~~~~~~~~~~~~~~~~~~

GiGaRunActionBase.h inherits from G4UserRunAction 

::

    [blyth@cms01 ~]$ find $DYB/NuWa-trunk/lhcb/Sim -name 'GiGa*ActionBase.h'
    /data/env/local/dyb/trunk/NuWa-trunk/lhcb/Sim/GiGa/GiGa/GiGaStepActionBase.h
    /data/env/local/dyb/trunk/NuWa-trunk/lhcb/Sim/GiGa/GiGa/GiGaEventActionBase.h
    /data/env/local/dyb/trunk/NuWa-trunk/lhcb/Sim/GiGa/GiGa/GiGaTrackActionBase.h
    /data/env/local/dyb/trunk/NuWa-trunk/lhcb/Sim/GiGa/GiGa/GiGaRunActionBase.h
    /data/env/local/dyb/trunk/NuWa-trunk/lhcb/Sim/GiGa/GiGa/GiGaStackActionBase.h

::

     26 class GiGaRunActionBase :
     27   public virtual IGiGaRunAction ,
     28   public          GiGaBase
     29 {


     30 class IGiGaRunAction:
     31   virtual public G4UserRunAction ,
     32   virtual public IGiGaInterface
     33 {



`source/run/include/G4UserRunAction.hh`::

     37 //  This is the base class of a user's action class which defines the
     38 // user's action at the begining and the end of each run. The user can
     39 // override the following two methods but the user should not change 
     40 // any of the contents of G4Run object.
     41 //    virtual void BeginOfRunAction(const G4Run* aRun);
     42 //    virtual void EndOfRunAction(const G4Run* aRun);
     43 // The user can override the following method to instanciate his/her own
     44 // concrete Run class. G4Run has a virtual method RecordEvent, so that
     45 // the user can store any information useful to him/her with event statistics.
     46 //    virtual G4Run* GenerateRun();
     47 //  The user's concrete class derived from this class must be set to
     48 // G4RunManager via G4RunManager::SetUserAction() method.
     49 //
     50 #include "G4Types.hh"
     51 
     52 class G4UserRunAction
     53 {
     54   public:
     55     G4UserRunAction();
     56     virtual ~G4UserRunAction();
     57 
     58   public:
     59     virtual G4Run* GenerateRun();
     60     virtual void BeginOfRunAction(const G4Run* aRun);
     61     virtual void EndOfRunAction(const G4Run* aRun);
     62 



GiGaRunActionExport
---------------------

`/data1/env/local/dyb/NuWa-trunk/lhcb/Sim/GaussTools/src/Components/GiGaRunActionExport.h`::


     28 class GiGaRunActionExport: public virtual GiGaRunActionBase
     29 {
     30   /// friend factory for instantiation
     31   //friend class GiGaFactory<GiGaRunActionExport>;
     32 
     33 public:
     34 
     35   typedef std::vector<G4VPhysicalVolume*> PVStack_t;
     36 
     37 
     38   /** performe the action at the begin of each run 
     39    *  @param run pointer to Geant4 run object 
     40    */
     41   void BeginOfRunAction ( const G4Run* run );
     42 
     43   /** performe the action at the end  of each event 
     44    *  @param run pointer to Geant4 run object 
     45    */
     46   void EndOfRunAction   ( const G4Run* run );
     47 

::

    660 void GiGaRunActionExport::BeginOfRunAction( const G4Run* run )
    661 {
    662 
    663   if( 0 == run )
    664     { Warning("BeginOfRunAction:: G4Run* points to NULL!") ; }
    665 
    666    G4VPhysicalVolume* wpv = G4TransportationManager::GetTransportationManager()->
    667       GetNavigatorForTracking()->GetWorldVolume();
    668 
    669 




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





