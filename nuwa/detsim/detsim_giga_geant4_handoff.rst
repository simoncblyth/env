DetSim GiGa Geant4 Handoff
============================

External Propagation Approaches
--------------------------------

collect and kill tracks, create hit collections
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#. in `DsChromaStackAction::ClassifyNewTrack` 

   * collect optical photon G4Track info into ChromaPhotonList member of StackAction
     and kill the G4Tracks, avoiding memory expense

#. in `DsChromaStackAction::NewStage` 

   * send the ChromaPhotonList off for external propagation, 
   * wait for response whilst GPU propagation proceeds 
     (need to arrange for standard PMTID to get thru to the GPU)
   * from the propagated photon response construct hit collection

#. in `DsChromaPullEvent::execute` OR  `DsPullEvent::execute` (if standard one can be used)

   * access the hit collection and send the hits along 
     as normally as possible


DsPmtSensDet::Initialize create HC for each (site,det) for each event
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
::

    195 void DsPmtSensDet::Initialize(G4HCofThisEvent* hce)
    196 {
    197     m_hc.clear();
    198 
    199     G4DhHitCollection* hc = new G4DhHitCollection(SensitiveDetectorName,collectionName[0]);
    200     m_hc[0] = hc;
    201     int hcid = G4SDManager::GetSDMpointer()->GetCollectionID(hc);
    202     hce->AddHitsCollection(hcid,hc);
    203 
    204     for (int isite=0; site_ids[isite] >= 0; ++isite) {
    205         for (int idet=0; detector_ids[idet] >= 0; ++idet) {
    206             DayaBay::Detector det(site_ids[isite],detector_ids[idet]);
    207 
    208             if (det.bogus()) continue;
    209 
    210             string name=det.detName();
    211             G4DhHitCollection* hc = new G4DhHitCollection(SensitiveDetectorName,name.c_str());
    212             short int id = det.siteDetPackedData();
    213             m_hc[id] = hc;
    214 
    215             int hcid = G4SDManager::GetSDMpointer()->GetCollectionID(hc);
    216             hce->AddHitsCollection(hcid,hc);
    217             debug() << "Add hit collection with hcid=" << hcid << ", cached ID="
    218                     << (void*)id
    219                     << " name= \"" << SensitiveDetectorName << "/" << name << "\""
    220                     << endreq;
    221         }
    222     }
    223 
    224     debug() << "DsPmtSensDet Initialize, made "
    225            << hce->GetNumberOfCollections() << " collections"
    226            << endreq;
    227    
    228 }


DsPmtSensDet::ProcessHits HC population
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* `step -> preStepPoint -> touchableHistory -> DetectorElement -> SensDetId`

* where are `SensDetId` associated with `DetectorElement` ?
* how do the `DetectorElement` and `touchableHistory` correspond to PVs ?

::

    318 bool DsPmtSensDet::ProcessHits(G4Step* step,
    319                                G4TouchableHistory* /*history*/)
    320 {
    321     //if (!step) return false; just crash for now if not defined
    322 
    323     // Find out what detector we are in (ADx, IWS or OWS)
    324     G4StepPoint* preStepPoint = step->GetPreStepPoint();
    325 
    326     double energyDep = step->GetTotalEnergyDeposit();
    327 
    328     if (energyDep <= 0.0) {
    329         //debug() << "Hit energy too low: " << energyDep/CLHEP::eV << endreq;
    330         return false;
    331     }
    332 
    333     const G4TouchableHistory* hist =
    334         dynamic_cast<const G4TouchableHistory*>(preStepPoint->GetTouchable());
    335     if (!hist or !hist->GetHistoryDepth()) {
    336         error() << "ProcessHits: step has no or empty touchable history" << endreq;
    337         return false;
    338     }
    339 
    340     const DetectorElement* de = this->SensDetElem(*hist);
    341     if (!de) return false;
    342 
    343     // wangzhe QE calculation starts here.
    344     int pmtid = this->SensDetId(*de);
    345     DayaBay::Detector detector(pmtid);
    ...
    ...
    231 const DetectorElement* DsPmtSensDet::SensDetElem(const G4TouchableHistory& hist)
    232 {
    233     const IDetectorElement* idetelem = 0;
    234     int steps=0;
    235 
    236     if (!hist.GetHistoryDepth()) {
    237         error() << "DsPmtSensDet::SensDetElem given empty touchable history" << endreq;
    238         return 0;
    239     }
    240 
    241     StatusCode sc =
    242         m_t2de->GetBestDetectorElement(&hist,m_sensorStructures,idetelem,steps);
    243     if (sc.isFailure()) {      // verbose warning
    244         warning() << "Failed to find detector element in:\n";
    245         for (size_t ind=0; ind<m_sensorStructures.size(); ++ind) {
    246             warning() << "\t\t" << m_sensorStructures[ind] << "\n";
    247         }
    248         warning() << "\tfor touchable history:\n";
    249         for (int ind=0; ind < hist.GetHistoryDepth(); ++ind) {
    250             warning() << "\t (" << ind << ") "
    251                       << hist.GetVolume(ind)->GetName() << "\n";
    252         }
    253         warning() << endreq;
    254         return 0;
    255     }
    256 
    257     return dynamic_cast<const DetectorElement*>(idetelem);
    258 }
    ...
    ...   //
    ...   // recurse up DetectorElement heirarchy until find an idParameter to return
    ...   // where are these int ID set ?  
    ...   //     * presumably generated by GiGaCnv 
    ...   //
    ...
    260 int  DsPmtSensDet::SensDetId(const DetectorElement& de)
    261 {
    262     const DetectorElement* detelem = &de;
    263 
    264     while (detelem) {
    265         if (detelem->params()->exists(m_idParameter)) {
    266             break;
    267         }
    268         detelem = dynamic_cast<const DetectorElement*>(detelem->parentIDetectorElement());
    269     }
    270     if (!detelem) {
    271         warning() << "Could not get PMT detector element starting from " << de << endreq;
    272         return 0;
    273     }
    274 
    275     return detelem->params()->param<int>(m_idParameter);
    276 }




GetTouchable
--------------

::

    delta:geant4.10.00.p01 blyth$ find . -name '*.hh' -exec grep -H GetTouchable {} \;
    ./source/parameterisations/gflash/include/G4GFlashSpot.hh:    G4TouchableHandle GetTouchableHandle() const {return theHandle;}
    ./source/parameterisations/gflash/include/G4VGFlashSensitiveDetector.hh:            tmpPoint->SetTouchableHandle(aSpot->GetTouchableHandle());
    ./source/track/include/G4ParticleChangeForLoss.hh:  aTrack->SetTouchableHandle(currentTrack->GetTouchableHandle());
    ./source/track/include/G4ParticleChangeForTransport.hh:    const G4TouchableHandle& GetTouchableHandle() const;
    ./source/track/include/G4StepPoint.hh:   const G4VTouchable* GetTouchable() const;
    ./source/track/include/G4StepPoint.hh:   const G4TouchableHandle& GetTouchableHandle() const;
    ./source/track/include/G4Track.hh:   const G4VTouchable*      GetTouchable() const;
    ./source/track/include/G4Track.hh:   const G4TouchableHandle& GetTouchableHandle() const;
    ./source/tracking/include/G4SteppingManager.hh:   const G4TouchableHandle& GetTouchableHandle();
    ./source/tracking/include/G4SteppingManager.hh:  inline const G4TouchableHandle& G4SteppingManager::GetTouchableHandle() {
    delta:geant4.10.00.p01 blyth$ 



HC Creation
------------

::

    [blyth@belle7 dybgaudi]$ find . -name '*.cc' -exec grep -H G4DhHitCollection {} \;
    ./Simulation/DetSim/src/DsPmtSensDet.cc:    G4DhHitCollection* hc = new G4DhHitCollection(SensitiveDetectorName,collectionName[0]);
    ./Simulation/DetSim/src/DsPmtSensDet.cc:            G4DhHitCollection* hc = new G4DhHitCollection(SensitiveDetectorName,name.c_str());
    ./Simulation/DetSim/src/DsPmtSensDet.cc:    G4DhHitCollection* hc = m_hc[sdid];
    ./Simulation/DetSim/src/DsRpcSensDet.cc:    G4DhHitCollection* hc = new G4DhHitCollection(SensitiveDetectorName,collectionName[0]);
    ./Simulation/DetSim/src/DsRpcSensDet.cc:            G4DhHitCollection* hc = new G4DhHitCollection(SensitiveDetectorName,name.c_str());
    ./Simulation/DetSim/src/DsRpcSensDet.cc:    G4DhHitCollection* hc = m_hc[sdid];
    ./Simulation/DetSim/src/DsPullEvent.cc:        G4DhHitCollection* g4hc = dynamic_cast<G4DhHitCollection*>(hcs->GetHC(ihc));
    ./Simulation/Fifteen/DetSimProc/src/DetSimProc.cc:  G4DhHitCollection* g4hc = dynamic_cast<G4DhHitCollection*>(hcs->GetHC(ihc));



Watershed : DsPullEvent
-------------------------

* watershed between python/pyroot/Gaudi/GiGa and underlying Geant4 at **DsPullEvent**


Stack Trace during propagation
--------------------------------

::

    513 
    514     494           operator[](size_type __n) const
    515     (gdb) bt
    516     #0  0x041f811a in std::vector<G4NavigationLevel, std::allocator<G4NavigationLevel> >::operator[] (this=0xc4045f4, __n=12) at /usr/lib/gcc/i386-redhat-linux/4.1.2/../.    ./../../include/c++/4.1.2/bits/stl_vector.h:494
    517     #1  0x041f81a3 in G4NavigationHistory::GetTopTransform (this=0xc4045f4) at /data1/env/local/dyb/NuWa-trunk/../external/build/LCG/geant4.9.2.p01/source/geometry/volume    s/include/G4NavigationHistory.icc:102
    518     #2  0x0703aa3c in G4Navigator::ComputeLocalAxis (this=0xc4045e8, pVec=@0xbfd17220) at include/G4Navigator.icc:57
    519     #3  0x070365cb in G4Navigator::ComputeStep (this=0xc4045e8, pGlobalpoint=@0xbfd17208, pDirection=@0xbfd17220, pCurrentProposedStepLength=47809528.913293302, pNewSafet    y=@0xbfd17238) at src/G4Navigator.cc:628
    520     #4  0x04e096fa in G4Transportation::AlongStepGetPhysicalInteractionLength (this=0xc06d4e8, track=@0x10a5a5c8, currentMinimumStep=47809528.913293302, currentSafety=@0x    bfd173b8, selection=0xc4042fc) at src/G4Transportation.cc:225
    521     #5  0x06e23e1b in G4VProcess::AlongStepGPIL (this=0xc06d4e8, track=@0x10a5a5c8, previousStepSize=17.522238749144233, currentMinimumStep=47809528.913293302, proposedSa    fety=@0xbfd173b8, selection=0xc4042fc)
    522         at /data1/env/local/dyb/NuWa-trunk/../external/build/LCG/geant4.9.2.p01/source/processes/management/include/G4VProcess.hh:447
    523     #6  0x06e22849 in G4SteppingManager::DefinePhysicalStepLength (this=0xc4041f0) at src/G4SteppingManager2.cc:235
    524     #7  0x06e1ee2c in G4SteppingManager::Stepping (this=0xc4041f0) at src/G4SteppingManager.cc:181
    525     #8  0x06e2d50a in G4TrackingManager::ProcessOneTrack (this=0xc4041c8, apValueG4Track=0x10a5a5c8) at src/G4TrackingManager.cc:126
    526     #9  0x06ea024f in G4EventManager::DoProcessing (this=0xc4039d8, anEvent=0x102ccca8) at src/G4EventManager.cc:185
    527     #10 0x06ea09e6 in G4EventManager::ProcessOneEvent (this=0xc4039d8, anEvent=0x102ccca8) at src/G4EventManager.cc:335
    528     #11 0xb4d2b5e8 in GiGaRunManager::processTheEvent (this=0xc403170) at ../src/component/GiGaRunManager.cpp:207
    529     #12 0xb4d2a522 in GiGaRunManager::retrieveTheEvent (this=0xc403170, event=@0xbfd17cf8) at ../src/component/GiGaRunManager.cpp:158
    530     #13 0xb4d0664f in GiGa::retrieveTheEvent (this=0xc402778, event=@0xbfd17cf8) at ../src/component/GiGa.cpp:469
    531     #14 0xb4d03564 in GiGa::operator>> (this=0xc402778, event=@0xbfd17cf8) at ../src/component/GiGaIGiGaSvc.cpp:73
    532     #15 0xb4d012fa in GiGa::retrieveEvent (this=0xc402778, event=@0xbfd17cf8) at ../src/component/GiGaIGiGaSvc.cpp:211
    533     #16 0xb4f4acd3 in DsPullEvent::execute (this=0xc3f5d00) at ../src/DsPullEvent.cc:54
    534     #17 0x069c1408 in Algorithm::sysExecute (this=0xc3f5d00) at ../src/Lib/Algorithm.cpp:558
    535     #18 0x0350ed4e in DybBaseAlg::sysExecute (this=0xc3f5d00) at ../src/lib/DybBaseAlg.cc:53
    536     #19 0x02cc6fd4 in GaudiSequencer::execute (this=0xbeb8140) at ../src/lib/GaudiSequencer.cpp:100
    537     #20 0x069c1408 in Algorithm::sysExecute (this=0xbeb8140) at ../src/Lib/Algorithm.cpp:558
    538     #21 0x02c5e68f in GaudiAlgorithm::sysExecute (this=0xbeb8140) at ../src/lib/GaudiAlgorithm.cpp:161
    539     #22 0x06a3d41a in MinimalEventLoopMgr::executeEvent (this=0xba77900) at ../src/Lib/MinimalEventLoopMgr.cpp:450
    540     #23 0x038ba956 in DybEventLoopMgr::executeEvent (this=0xba77900, par=0x0) at ../src/DybEventLoopMgr.cpp:125
    541     #24 0x038bb18a in DybEventLoopMgr::nextEvent (this=0xba77900, maxevt=1) at ../src/DybEventLoopMgr.cpp:188
    542     #25 0x06a3bdbd in MinimalEventLoopMgr::executeRun (this=0xba77900, maxevt=1) at ../src/Lib/MinimalEventLoopMgr.cpp:400
    543     #26 0x093096d9 in ApplicationMgr::executeRun (this=0xb744aa0, evtmax=1) at ../src/ApplicationMgr/ApplicationMgr.cpp:867
    544     #27 0x0829bf57 in method_3426 (retaddr=0xc4f7d00, o=0xb744ecc, arg=@0xb7b0c20) at ../i686-slc5-gcc41-dbg/dict/GaudiKernel/dictionary_dict.cpp:4375
    545     #28 0x001d6add in ROOT::Cintex::Method_stub_with_context (context=0xb7b0c18, result=0xc53d26c, libp=0xc53d2c4) at cint/cintex/src/CINTFunctional.cxx:319
    546     #29 0x0330e034 in ?? ()
    547     #30 0x0b7b0c18 in ?? ()
    548     #31 0x0c53d26c in ?? ()
    549     #32 0x00000000 in ?? ()
    550     Current language:  auto; currently c++
    551     (gdb) 




DsPullEvent
-----------


`NuWa-trunk/dybgaudi/Simulation/DetSim/src/DsPullEvent.cc`::

     40 StatusCode DsPullEvent::execute()
     41 {
     42     DayaBay::SimHeader* header = MakeHeaderObject();
     43 
     44     // Just pass through GenHeader's timestamp.  This also causes
     45     // GenHeader to be registered as input, something that would
     46     // normally just happen if DsPushKine and DsPullEvent were the
     47     // same algorithm.
     48     DayaBay::GenHeader* gen_header = getTES<DayaBay::GenHeader>(m_genLocation);
     49     header->setTimeStamp(gen_header->timeStamp());
     50 
     51     //////////////////////////
     52     // Primary event vertices.
     53     const G4Event* g4event = 0;
     54     m_giga->retrieveEvent(g4event);
     55     if (!g4event) {
     56         error() << "No G4Event!" << endreq;
     57         return StatusCode::FAILURE;
     58     }
     59 
     60     // reset Capture
     61     G4DhNeutronCapture capture;
     62     m_capinfo->addCapture(capture);
     63 
     64     int nverts = g4event->GetNumberOfPrimaryVertex();
     65     if( nverts == 0 ) {
     66         warning() << "The g4event has zero primary vertices!" << endreq;
     67         return StatusCode::SUCCESS;
     68     }
     69 
     70 
     71     debug() << "Pulled event with " << nverts
     72            << " primary vertices, event id:" << g4event->GetEventID() << endreq;
     73     G4PrimaryVertex* g4vtx = g4event->GetPrimaryVertex(0);
     74     while (g4vtx) {
     75         debug() << "\n\tat (" << g4vtx->GetX0() << "," << g4vtx->GetY0() << "," << g4vtx->GetZ0() << ")";
     76         g4vtx = g4vtx->GetNext();
     77         break;
     78     }
     79     debug() << endreq;
     80 
     81     //////////////////////////
     82     // particle histories.
     83     // Do this first so we can use it below.
     84     DayaBay::SimParticleHistory* history =0;
     85     m_historyKeeper->ClaimCurrentHistory(history); // This takes ownership from the Keeper.
     86     header->setParticleHistory(history);
     87 
     88     //////////////////////////
     89     // Unobservable Statistics
     90     DayaBay::SimUnobservableStatisticsHeader* unobs =0;
     91     m_historyKeeper->ClaimCurrentUnobservable(unobs); // This takes ownership from the Keeper.
     92     header->setUnobservableStatistics(unobs);
     93 
     94     //////////////////////////
     95     // Hit collections.
     96     G4HCofThisEvent* hcs = g4event->GetHCofThisEvent();
     97     if (!hcs) {
     98         warning() << "No HitCollections in this event" << endreq;
     99         return StatusCode::SUCCESS;
     00     }
     01     int nhc = hcs->GetNumberOfCollections();
     02     if (!nhc) {
     03         warning() << "Number of HitCollections is zero" << endreq;
     04         return StatusCode::SUCCESS;
     05     }
     06     debug () << "# HitCollections = " << nhc << endreq;
     07 
     08     // introduce the headers to each other
     09     DayaBay::SimHitHeader* hit_header = new DayaBay::SimHitHeader(header);
     10     header->setHits(hit_header);
     11 
     12     double earliestTime = 0;
     13     double latestTime = 0;
     14     Context context;
     15     context.SetSimFlag(SimFlag::kMC);
     16     bool firstDetector = true;
     17     int hitcount=0;  // deal with no hits situation
     18 
     19     for (int ihc=0; ihc<nhc; ++ihc) {
     20         G4DhHitCollection* g4hc = dynamic_cast<G4DhHitCollection*>(hcs->GetHC(ihc));
     21         if (!g4hc) {
     22             error() << "Failed to get hit collection #" << ihc << endreq;
     23             return StatusCode::FAILURE;
     24         }
     25 
     26         // DetSim produces hit collections even for unsimulated detectors
     27         size_t nhits = g4hc->GetSize();
     28     hitcount+=nhits;
     29         if (!nhits) continue;
     30 
     31     bool firstHit = true;
     32         DayaBay::SimHitCollection::hit_container hits;
     33     DayaBay::Detector detector;
     34         DayaBay::SimHitCollection* shc =
     35       new DayaBay::SimHitCollection(hit_header,detector,hits);
     36         for (size_t ihit=0; ihit<nhits; ++ihit) {

