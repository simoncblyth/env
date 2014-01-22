Chroma NuWa Integration
==========================

Thoughts
---------

GPU hit creation maybe a step too far, as stepping too close to areas of detector custom code.

   * perhaps just transport the OPs and hand back to G4 as about to hit sensitive detectors ?
   * OR creating hits (times and charges)

Look into the nature of G4/DetSim "hits", how difficult to take it to that level ?
Wherever the handover happens, the approach to material/solid identity on the mesh needs to be understood

Geant4/DetSim reminder
------------------------

::

    (gdb) b 'DsPmtSensDet::ProcessHits(G4Step*, G4TouchableHistory*)' 
    Breakpoint 2 at 0xb472d1f8: file ../src/DsPmtSensDet.cc, line 324.


* works out the volume, determines QE, knarly QE diddling, hit creation   

   * not impossible to duplicate, 
   * BUT disadvantage is mixing detector specific diddling with OP acceleration
   
       * maybe split stage, to keep detector specifics separate 

::

    [blyth@cms01 src]$ pwd
    /data/env/local/dyb/trunk/NuWa-trunk/dybgaudi/Simulation/DetSim/src
    [blyth@cms01 src]$ grep ProcessHits *.cc
    DsPmtSensDet.cc:bool DsPmtSensDet::ProcessHits(G4Step* step,
    DsPmtSensDet.cc:        error() << "ProcessHits: step has no or empty touchable history" << endreq;
    DsRpcSensDet.cc:bool DsRpcSensDet::ProcessHits(G4Step* step,
    DsRpcSensDet.cc:      error() << "ProcessHits: step has no or empty touchable history." 
    [blyth@cms01 src]$ 


DsPmtModel, associating PMT names to volumes::

     04 #include "G4VFastSimulationModel.hh"
     05 
     06 class G4LogicalVolume;
     07 class G4Hooks;                  // opaque
     08     
     09 class DsPmtModel : public G4VFastSimulationModel
     10 {   
     11 public:
     12     DsPmtModel(const G4String& name, G4Envelope* volume,
     13                G4bool unique = false);
     14     DsPmtModel(const G4String& name);
     15     virtual ~DsPmtModel ();
     16     



Questions
------------

how/when to give OP tracks back to G4/reconstruction code ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Maybe **DsOpGPUStackAction**

#. collect OP into `fWaiting` stack (similar to DsOpStackAction)  
#. at `NewStage` 

  * make **interesting-or-not judgement**
  * translate OP G4Tracks into numpy arrays ready for Chroma/GPU  
  * perform OP cohort external propagation on GPU

     * where to stop GPU propagation ? defined SD volume ?  

  * translate back from numpy arrays diddling the waiting G4Tracks [where/access?]

     * `NewStage` invokes a reclassify `stackManager->ReClassify();` giving access
        to all the tracks in the *ClassifyNewTrack* allowing diddling then like the photon reweighting of
        `G4ClassificationOfNewTrack DsFastMuonStackAction::ClassifyNewTrack (const G4Track* aTrack)`

  * resume the G4 tracks by returning as `fUrgent`, which should immediately proceed into sens det handling 

     * how does SD/hit handover work /data/env/local/dyb/trunk/NuWa-trunk/dybgaudi/Simulation/DetSim/src/DsPmtSensDet.cc




how does G4 OP propagation end ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


translation of DYB solid geomety into surface tris
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



what about some  magic *optransport* physics process ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

NO, as need to deal with the OP as a cohort, not individually.
Physics processes act on individual OP.


OP collection and propagation, kicked off where ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* `G4ClassificationOfNewTrack DsOpStackAction::ClassifyNewTrack (const G4Track* aTrack)` 

   * assigns fWaiting status to OP, causing collection of OP tracks in the waiting stack 

* status is flipped to proceed with OP propagation only for events deemed to be interesting
* the judgement and kick-off happens in `void DsOpStackAction::NewStage()` which is invoked
  when the `fUrgent` stack is empty (ie everything other than the waiting tracks have been tracked) 
  and `fWaiting` stack has entries


* a similar structure seems good for GPU propagation

   * collect all the OP to benefit from massive parallelisation



