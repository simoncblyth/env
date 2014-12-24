
G4DAEChroma
=============

Objective
------------

Pull out everything Chroma related and reusable 
from StackAction and SensitiveDetector
for flexible reusability from different Geant4 contexts


Recent Changes
----------------


Major recent changes are:

* forming PMT hits on the GPU, by associating every triangle 
  of the PMT cathode with the PMT identifier. 
  This allows returning only photons that hit PMTs to Geant4

* transforming global hit coordinates into local coordinates for the hits using 
  a per PMT transform cache 

* integrating the GPU formed hits into the standard Geant4 hit collections, 
  allowing subsequent simulation to work unchanged

* moving Scintillation and Cerenkov generation to the GPU by 
  collecting generation steps and transferring those.
  Avoids having to transfer ~200MB of photon data for a 3M photon event :
  instead down to  ~2MB of generation steps.

  Its also advantageous never to even create G4Track for the millions
  of photons : they are skipped in the Scintillation and Cerenkov processes

* adopting NumPy serialization format, so can effectively directly fill
  NumPy arrays from Geant4 C++ 
  Allows to get rid of ROOT TObject, (and ROOT entirely) and annoying 
  dictionary generation hassles.

* for monitoring, control and communication  G4DAEChroma includes 
  integration with sqlite3 and a cJSON implementation allowing python
  dicts to be passed to C++ as map<string,string> which can be 
  inserted into sqlite3 tables 

* G4DAE geometry exporter was presented at the Geant4 Okinawa Collaboration 
  meeting in Sept. I proposed to contribute the exporter to Geant4 and they agreed.  
  The plan is to include it with the Geant4 released at the end of 2015.



TODO : Test on Workstation GPU
-------------------------------

So far I have been developing using my laptop GPU only. 
I think I will soon be interested in testing the machinery to see 
the kind of performance possible on workstation GPUs.

Comparing my laptop GPU (with that 

* http://www.techpowerup.com/gpudb/2527/geforce-gt-750m-mac-edition.html
* http://www.techpowerup.com/gpudb/2029/tesla-k20m.html



Dependencies
------------

* ROOT, CLHEP, Geant4, ZMQ, ZMQRoot
* **NOT** detsim/giga/gaudi/gauss

  * G4DAEChroma intended to be used from minimal 
    shim G4 Actions hooked up in DetSim/GiGa 
    in as few lines of code as possible

  * this facilities limited dependency development/testing 
    providing faster dev cycle : see `mocknuwa-` `datamodel-`


Current Approach Aiming at 
------------------------------

* Cerenkov/Scintillation processes generating large numbers of secondary Optical Photons
* these are put into track stack 
* collected from track stack and track killed into PhotonList
* PhotonList is serialized, sent over network and deseralized
* PhotonList then copied to GPU and propagated
* Photons then copied back from GPU
* hit selection is applied (ie need to have reached SURFACE_DETECT) and a HitList 
  is compiled
* HitList then serialized and replied to original request 
* HitList deserialized back in Geant4 process, and collected into 
  standard DsPmtSensDet hit collections using TrojanSensDet approach  



Ultimate photon list compression approach 
------------------------------------------

Moving generation of PhotonList to the other end of the transport
would allow to only transfer generation parameters 
of the PhotonList cohort as formed in the Cerenkov/Scintillation process. 

* **Ultimate Photon List compression : just transfer parameters of the cohort**


Transform cache problem
------------------------

* currently keyed on volume_index but that is not being trasported, only pmtid is 


Geometry Consistency between GPU and MEMORY trees
--------------------------------------------------

Geometry matching between in MEMORY G4 tree 
and the one reconstructed from DAE file should
be done in a handshake REQ/REP. 

This will be particularly useful when
handling multiple partial geometries.    

* maybe implement in some metadata properties of the transport object 


Appropriate identifiers
-----------------------------

Currently make heavy use of volume traverse indices.  
This is a frail approach as using partial geometries or otherwise updating 
geometry will change the indices.
Better to affix things like transforms to PMT Identifiers
which can be regarded as permanent identifiers.

Generalization of PMT Identifier handling
------------------------------------------

* **Construct Volume Transform cache keyed on PmtId**

Want to add IDMAP handling to G4DAEGeometry organized into 

* generic base, depending on Geant4 only

  * Geant4 volume traversal
  * making PVStack for all nodes of the tree 
  * forming TouchableHistories for all Sensitive Detectors 
  * a default implementation of providing identifiers could 
    just be the SD index in volume traverse order

* Dyb specialized level, depending on Gaudi/Gauss/DetectorElements/NuWa:TH2DE 

  * yields PmtId from TouchableHistory for all SD, 


Incorporate IDMAP Handling into G4DAEChroma/G4DAEGeometry
----------------------------------------------------------

#. Chroma GPU photons reaching SURFACE_DETECT know the solid/volume index they are landing on
#. to be able to return only hits the association from volume index  
   a PmtId is needed in order to form a **hit**

Need to know the PmtId at GPU level, so have to propagate this forward.

The PmtId are associated to pieces of geometry "DetectorElements"
at Gaudi level, this info does not exist at Geant4 level, so have 
to do contortions to make the association of volume traverse
index and pmtid.

Merge what is currently done separately in the below into G4DAEGeometry

* NuWa-trunk/lhcb/Sim/GaussTools/src/Components/GiGaRunActionExport.h
* NuWa-trunk/lhcb/Sim/GaussTools/src/Components/GiGaRunActionExport.cpp

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



Trace channel_id/PmtId from CUDA kernel backwards
----------------------------------------------------------------

* a positive `last_hit_triangle` for the `SURFACE_DETECT` subset indicates a **hit** 

`chroma/chroma/cuda/propagate_hit.cu`::

    118 // iiPPPPPPPPPPPiiiP
    119 
    120 __global__ void
    121 propagate_hit(
    122       int first_photon,
    123       int nthreads,
    124       unsigned int *input_queue,
    125       unsigned int *output_queue,
    126       curandState *rng_states,
    127       float3 *positions,
    128       float3 *directions,
    129       float *wavelengths,
    130       float3 *polarizations,
    131       float *times,
    132       unsigned int *histories,
    133       int *last_hit_triangles,
    134       float *weights,
    135       int max_steps,
    136       int use_weights,
    137       int scatter_first,
    138       Geometry *g,
    139       int* solid_map,
    140       int* solid_id_to_channel_id )
    141 {
    142     __shared__ Geometry sg;
    143 
    ...
    233     if ((p.history & SURFACE_DETECT) != 0) {
    234 
    235         //
    236         // kludgy mis-use of lht for outputting 
    237         // various things like 
    238         //       solid_id:    index like, zero based
    239         //       channel_id:  the pmtid, encoding site/ad/ring/...
    240         //
    241         int triangle_id = last_hit_triangles[photon_id];
    242         if (triangle_id > -1) {
    243             int solid_id = solid_map[triangle_id];
    244             int channel_id = solid_id_to_channel_id[solid_id];
    245             last_hit_triangles[photon_id] = channel_id ;
    246         } else {
    247             last_hit_triangles[photon_id] = -2 ;
    248         }
    249 
    250     }
    251 
    252 
    253 
    254 } // propagate_hit


`chroma/chroma/gpu/photon_hit.py`::

    154     def propagate_hit(self,
    155                   gpu_geometry,
    156                   rng_states,
    157                   nthreads_per_block=64,
    158                   max_blocks=1024,
    159                   max_steps=100,
    160                   use_weights=False,
    161                   scatter_first=0):
    162         """Propagate photons on GPU to termination or max_steps, whichever
    163         comes first.
    164 
    165         May be called repeatedly without reloading photon information if
    166         single-stepping through photon history.
    167 
    168         ..warning::
    169             `rng_states` must have at least `nthreads_per_block`*`max_blocks`
    170             number of curandStates.
    171         """
    172         nphotons = self.pos.size
    173         nwork = nphotons
    174         self.upload_queues( nwork )
    175 
    176         solid_id_map_gpu = gpu_geometry.solid_id_map
    177         solid_id_to_channel_id_gpu = gpu_geometry.solid_id_to_channel_id_gpu
    178 


Modified original GPUPhotons to GPUPhotonsHit to add this hit info.

`env/geant4/geometry/collada/g4daeview/daedirectpropagator.py`::

     21 from chroma.gpu.tools import get_cu_module, cuda_options, chunk_iterator, to_float3
     22 #from chroma.gpu.photon import GPUPhotons
     23 from chroma.gpu.photon_hit import GPUPhotonsHit
     24 from chroma.gpu.geometry import GPUGeometry
     ..
     27 class DAEDirectPropagator(object):
     28     def __init__(self, config, chroma):
     29         """
     30         :param config:
     31         :param chroma: DAEChromaContext instance 
     32         """
     33         self.config = config
     34         self.chroma = chroma
     35 
     36     def propagate(self, cpl, max_steps=100 ):
     37         """
     38         :param cpl: ChromaPhotonList instance
     39         :return propagated_cpl: ChromaPhotonList instance
     40 
     41         """
     42         photons = Photons.from_cpl(cpl, extend=True)  # CPL into chroma.event.Photons OR photons.Photons   
     43         gpu_photons = GPUPhotonsHit(photons)
     44         gpu_detector = self.chroma.gpu_detector
     45 
     46         gpu_photons.propagate_hit(gpu_detector,
     47                                   self.chroma.rng_states,
     48                                   nthreads_per_block=self.chroma.nthreads_per_block,
     49                                   max_blocks=self.chroma.max_blocks,
     50                                   max_steps=max_steps)
     51 
     52         photons_end = gpu_photons.get()
     53         self.photons_end = photons_end
     54         return create_cpl_from_photons_very_slowly(photons_end)



Copying mapping array to GPU.

`chroma/chroma/gpu/detector.py`::

     16 class GPUDetector(GPUGeometry):
     17     def __init__(self, detector, wavelengths=None, print_usage=False):
     18         GPUGeometry.__init__(self, detector, wavelengths=wavelengths, print_usage=False)
     19 
     20         self.solid_id_to_channel_index_gpu = \
     21             ga.to_gpu(detector.solid_id_to_channel_index.astype(np.int32))
     22         self.solid_id_to_channel_id_gpu = \
     23             ga.to_gpu(detector.solid_id_to_channel_id.astype(np.int32))
     24 
     25         self.nchannels = detector.num_channels()



Populate mapping array on **add_pmt**

`chroma/chroma/detector.py`::

     59     def add_pmt(self, pmt, rotation=None, displacement=None, channel_id=None):
     60         """Add the PMT `pmt` to the geometry. When building the final triangle
     61         mesh, `solid` will be placed by rotating it with the rotation matrix
     62         `rotation` and displacing it by the vector `displacement`, just like
     63         add_solid().
     64 
     65             `pmt``: instance of chroma.Solid
     66                 Solid representing a PMT.
     67             `rotation`: numpy.matrix (3x3)
     68                 Rotation to apply to PMT mesh before displacement.  Defaults to
     69                 identity rotation.
     70             `displacement`: numpy.ndarray (shape=3)
     71                 3-vector displacement to apply to PMT mesh after rotation.
     72                 Defaults to zero vector.
     73             `channel_id`: int
     74                 Integer identifier for this PMT.  May be any integer, with no
     75                 requirement for consective numbering.  Defaults to None,
     76                 where the ID number will be set to the generated channel index.
     77                 The channel_id must be representable as a 32-bit integer.
     78         
     79             Returns: dictionary { 'solid_id' : solid_id, 
     80                                   'channel_index' : channel_index,
     81                                   'channel_id' : channel_id }
     82         """
     83 
     84         solid_id = self.add_solid(solid=pmt, rotation=rotation,
     85                                   displacement=displacement)
     86 
     87         channel_index = len(self.channel_index_to_solid_id)
     88         if channel_id is None:
     89             channel_id = channel_index
     90 
     91         # add_solid resized this array already
     92         self.solid_id_to_channel_index[solid_id] = channel_index
     93         self.solid_id_to_channel_id[solid_id] = channel_id
     94 


`channel_id` Identifiers are affixed to the DAENode.

`env/geant4/geometry/collada/collada_to_chroma.py`::

    701         solid = Solid( mesh, material1, material2, surface, color )
    702         solid.node = node
    703 
    704         #
    705         # hmm a PMT is comprised of several volumes all of which 
    706         # have the same associated channel_id 
    707         #
    708         channel_id = getattr(node, 'channel_id', None)
    709         if not channel_id is None and channel_id > 0:
    710             self.channel_count += 1             # nodes with associated non zero channel_id
    711             self.channel_ids.add(channel_id)
    712             self.chroma_geometry.add_pmt( solid, channel_id=channel_id)
    713         else:
    714             self.chroma_geometry.add_solid( solid )
    715         pass



This is relying on the **idmap** (keyed by volume index) that is 
written by `GiGaRunActionExport`

* NuWa-trunk/lhcb/Sim/GaussTools/src/Components/GiGaRunActionExport.cpp


`env/geant4/geometry/collada/g4daenode.py`::

     375     @classmethod
     376     def idmaplink(cls, idmap ):
     377         if idmap is None:
     378             log.warn("skip idmaplink ")
     379             return
     380         pass
     381         log.info("linking DAENode with idmap %s identifiers " % len(idmap))
     382         assert len(cls.registry) == len(idmap), ( len(cls.registry), len(idmap))
     383         for index, node in enumerate(cls.registry):
     384             node.channel_id = idmap[index]
     385             node.g4tra = idmap.tra[index]
     386             node.g4rot = idmap.rot[index]
     387             #if index % 100 == 0:
     388             #    print index, node.channel_id, node, node.__class__
     389 



