VRML2FILE implementation
==========================

.. contents:: :local:

Observations
-------------

The VRML2 polyhedra created, have either triangle or quad faces ?

* how to switch to triangle ? or just fix by splitting quad afterwards ?


Visualization Steering
----------------------

stacktrace from draw command to graphics primitives
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    (gdb) b 'G4VRML2FileSceneHandler::AddPrimitive(G4Polyhedron const&)' 
    Breakpoint 1 at 0x46c7fcf: file src/G4VRML2SceneHandlerFunc.icc, line 175.
    (gdb) 
    ...
    /vis/sceneHandler/attach
    Scene "scene-0" attached to scene handler "scene-handler-0.
      (You may have to refresh with "/vis/viewer/flush" if view is not "auto-refresh".)
    NOTE: For systems which are not "auto-refresh" you will need to
      issue "/vis/viewer/refresh" or "/vis/viewer/flush".
    /vis/viewer/refresh viewer-0
    Refreshing viewer "viewer-0 (VRML2FILE)"...
    ===========================================
    Output VRML 2.0 file: g4_03.wrl
    Maximum number of files in the destination directory: 100
      (Customizable with the environment variable: G4VRMLFILE_MAX_FILE_NUM) 
    ===========================================
    Traversing scene data...

    Breakpoint 1, G4VRML2FileSceneHandler::AddPrimitive (this=0x1024ab10, polyhedron=@0x1024aaa8) at src/G4VRML2SceneHandlerFunc.icc:175
    175             if (polyhedron.GetNoFacets() == 0) return;
    (gdb) bt
    #0  G4VRML2FileSceneHandler::AddPrimitive (this=0x1024ab10, polyhedron=@0x1024aaa8) at src/G4VRML2SceneHandlerFunc.icc:175
    #1  0xb709af73 in G4VSceneHandler::RequestPrimitives (this=0x1024ab10, solid=@0xbb7cce0) at src/G4VSceneHandler.cc:453
    #2  0xb709999b in G4VSceneHandler::AddSolid (this=0x1024ab10, solid=@0xbb7cce0) at src/G4VSceneHandler.cc:252
    #3  0x046c8889 in G4VRML2FileSceneHandler::AddSolid (this=0x1024ab10, vsolid=@0xbb7cce0) at src/G4VRML2SceneHandlerFunc.icc:78
    #4  0x0458f405 in G4SubtractionSolid::DescribeYourselfTo (this=0xbb7cce0, scene=@0x1024ab10) at src/G4SubtractionSolid.cc:459
    #5  0xb5ef6186 in G4PhysicalVolumeModel::DescribeSolid (this=0x1024b0c0, theAT=@0xbfabb230, pSol=0xbb7cce0, pVisAttribs=0x1024cdc8, sceneHandler=@0x1024ab10) at src/G4PhysicalVolumeModel.cc:551
    #6  0xb5ef7bf2 in G4PhysicalVolumeModel::DescribeAndDescend (this=0x1024b0c0, pVPV=0xba26fd0, requestedDepth=-2, pLV=0xbb61578, pSol=0xbb7cce0, pMaterial=0xbb7e670, theAT=@0xbfabba10, sceneHandler=@0x1024ab10) at src/G4PhysicalVolumeModel.cc:456
    #7  0xb5ef6ba3 in G4PhysicalVolumeModel::VisitGeometryAndGetVisReps (this=0x1024b0c0, pVPV=0xba26fd0, requestedDepth=-2, theAT=@0xbfabba10, sceneHandler=@0x1024ab10) at src/G4PhysicalVolumeModel.cc:222
    #8  0xb5ef7dc1 in G4PhysicalVolumeModel::DescribeAndDescend (this=0x1024b0c0, pVPV=0xbc0f528, requestedDepth=-1, pLV=0xba26c80, pSol=0xba26c00, pMaterial=0xbe623f8, theAT=@0xbfabbf80, sceneHandler=@0x1024ab10) at src/G4PhysicalVolumeModel.cc:523
    #9  0xb5ef6ba3 in G4PhysicalVolumeModel::VisitGeometryAndGetVisReps (this=0x1024b0c0, pVPV=0xbc0f528, requestedDepth=-1, theAT=@0xbfabbf80, sceneHandler=@0x1024ab10) at src/G4PhysicalVolumeModel.cc:222
    #10 0xb5ef8219 in G4PhysicalVolumeModel::DescribeYourselfTo (this=0x1024b0c0, sceneHandler=@0x1024ab10) at src/G4PhysicalVolumeModel.cc:170
    #11 0xb709a9f2 in G4VSceneHandler::ProcessScene (this=0x1024ab10) at src/G4VSceneHandler.cc:516
    #12 0xb70a162b in G4VViewer::ProcessView (this=0x1024cc58) at src/G4VViewer.cc:122
    #13 0x046c980c in G4VRML2FileViewer::DrawView (this=0x1024cc58) at src/G4VRML2FileViewer.cc:83
    #14 0xb70d81d7 in G4VisCommandViewerRefresh::SetNewValue (this=0xbf8a760, newValue=
              {<std::basic_string<char,std::char_traits<char>,std::allocator<char> >> = {static npos = 4294967295, _M_dataplus = {<std::allocator<char>> = {<__gnu_cxx::new_allocator<char>> = {<No data fields>}, <No data fields>}, _M_p = 0xbfabc1d8 "o2$\020\001"}}, <No data fields>}) at src/G4VisCommandsViewer.cc:1141
    #15 0x072c1dc4 in G4UIcommand::DoIt (this=0xbf8b010, parameterList=
              {<std::basic_string<char,std::char_traits<char>,std::allocator<char> >> = {static npos = 4294967295, _M_dataplus = {<std::allocator<char>> = {<__gnu_cxx::new_allocator<char>> = {<No data fields>}, <No data fields>}, _M_p = 0xbfabc28c "$2$\020\023"}}, <No data fields>}) at src/G4UIcommand.cc:210
    #16 0x072ce32b in G4UImanager::ApplyCommand (this=0xbe49a98, aCmd=0x1024aeec "/vis/viewer/refresh viewer-0") at src/G4UImanager.cc:410
    #17 0x072ce47a in G4UImanager::ApplyCommand (this=0xbe49a98, aCmd=
              {<std::basic_string<char,std::char_traits<char>,std::allocator<char> >> = {static npos = 4294967295, _M_dataplus = {<std::allocator<char>> = {<__gnu_cxx::new_allocator<char>> = {<No data fields>}, <No data fields>}, _M_p = 0xbfabc2f8 "i(R)$\020i(R)$\020THaN(w\r\005"}}, <No data fields>}) at src/G4UImanager.cc:354
    #18 0xb70d8adb in G4VisCommandViewerFlush::SetNewValue (this=0xbf89e98, newValue=
              {<std::basic_string<char,std::char_traits<char>,std::allocator<char> >> = {static npos = 4294967295, _M_dataplus = {<std::allocator<char>> = {<__gnu_cxx::new_allocator<char>> = {<No data fields>}, <No data fields>}, _M_p = 0xbfabc438 "d2$\020\001"}}, <No data fields>}) at src/G4VisCommandsViewer.cc:801
    #19 0x072c1dc4 in G4UIcommand::DoIt (this=0xbf8a028, parameterList=
              {<std::basic_string<char,std::char_traits<char>,std::allocator<char> >> = {static npos = 4294967295, _M_dataplus = {<std::allocator<char>> = {<__gnu_cxx::new_allocator<char>> = {<No data fields>}, <No data fields>}, _M_p = 0xbfabc4ec "1/4\233\200\004yyyy\021"}}, <No data fields>}) at src/G4UIcommand.cc:210
    #20 0x072ce32b in G4UImanager::ApplyCommand (this=0xbe49a98, aCmd=0xbf7075c "/vis/viewer/flush") at src/G4UImanager.cc:410
    #21 0x072ce47a in G4UImanager::ApplyCommand (this=0xbe49a98, aCmd=
              {<std::basic_string<char,std::char_traits<char>,std::allocator<char> >> = {static npos = 4294967295, _M_dataplus = {<std::allocator<char>> = {<__gnu_cxx::new_allocator<char>> = {<No data fields>}, <No data fields>}, _M_p = 0xbfabc5c8 "\\\a/\v\230\232a\vO\204\032\005~O]OAE<<?\022}\030\005h0o\vpC$\020 AE<<?O\226e\vAXja"}}, <No data fields>}) at src/G4UImanager.cc:354
    #22 0xb744c4f2 in GiGaRunActionCommand::BeginOfRunAction (this=0xbf63068, run=0x1024c770) at ../src/Components/GiGaRunActionCommand.cpp:79
    #23 0x05187d12 in G4RunManager::RunInitialization (this=0xbe9c014) at src/G4RunManager.cc:203
    #24 0xb45dd722 in GiGaRunManager::initializeRun (this=0xbe9be40) at ../src/component/GiGaRunManager.cpp:347
    #25 0xb45dd984 in GiGaRunManager::prepareTheEvent (this=0xbe9be40, vertex=0xbb24038) at ../src/component/GiGaRunManager.cpp:250
    #26 0xb45bae97 in GiGa::prepareTheEvent (this=0xbe9b448, vertex=0xbb24038) at ../src/component/GiGa.cpp:444
    #27 0xb45b7620 in GiGa::operator<< (this=0xbe9b448, vertex=0xbb24038) at ../src/component/GiGaIGiGaSvc.cpp:52
    #28 0xb4806ad5 in DsPushKine::execute (this=0xbe8d218) at ../src/DsPushKine.cc:55
    #29 0x0226c408 in Algorithm::sysExecute (this=0xbe8d218) at ../src/Lib/Algorithm.cpp:558
    #30 0x0601768f in GaudiAlgorithm::sysExecute (this=0xbe8d218) at ../src/lib/GaudiAlgorithm.cpp:161
    #31 0x0607ffd4 in GaudiSequencer::execute (this=0xb951210) at ../src/lib/GaudiSequencer.cpp:100
    #32 0x0226c408 in Algorithm::sysExecute (this=0xb951210) at ../src/Lib/Algorithm.cpp:558
    #33 0x0601768f in GaudiAlgorithm::sysExecute (this=0xb951210) at ../src/lib/GaudiAlgorithm.cpp:161
    #34 0x022e841a in MinimalEventLoopMgr::executeEvent (this=0xb50e180) at ../src/Lib/MinimalEventLoopMgr.cpp:450
    #35 0x03e12956 in DybEventLoopMgr::executeEvent (this=0xb50e180, par=0x0) at ../src/DybEventLoopMgr.cpp:125
    #36 0x03e1318a in DybEventLoopMgr::nextEvent (this=0xb50e180, maxevt=10) at ../src/DybEventLoopMgr.cpp:188
    #37 0x022e6dbd in MinimalEventLoopMgr::executeRun (this=0xb50e180, maxevt=10) at ../src/Lib/MinimalEventLoopMgr.cpp:400
    #38 0x090a46d9 in ApplicationMgr::executeRun (this=0xb1d4cb8, evtmax=10) at ../src/ApplicationMgr/ApplicationMgr.cpp:867
    #39 0x07021f57 in method_3426 (retaddr=0xbf9d3f0, o=0xb1d50e4, arg=@0xb240e30) at ../i686-slc5-gcc41-dbg/dict/GaudiKernel/dictionary_dict.cpp:4375
    #40 0x007ccadd in ROOT::Cintex::Method_stub_with_context (context=0xb240e28, result=0xbfe6224, libp=0xbfe627c) at cint/cintex/src/CINTFunctional.cxx:319
    #41 0x03921034 in ?? ()
    #42 0x0b240e28 in ?? ()
    #43 0x0bfe6224 in ?? ()
    #44 0x00000000 in ?? ()
    (gdb) 

    (gdb) frame 13
    #13 0x046c980c in G4VRML2FileViewer::DrawView (this=0x1024cc58) at src/G4VRML2FileViewer.cc:83
    83              ProcessView();
    (gdb) list
    78              // Viewpoint node
    79              SendViewParameters(); 
    80
    81              // Here is a minimal DrawView() function.
    82              NeedKernelVisit();
    83              ProcessView();
    84              FinishView();
    85      }
    86

    (gdb) frame 12
    #12 0xb70a162b in G4VViewer::ProcessView (this=0x1024cc58) at src/G4VViewer.cc:122
    122         fSceneHandler.ProcessScene (*this);
    (gdb) list
    117       // DrawView)...
    118       if (fNeedKernelVisit) {
    119         // Reset flag.  This must be done before ProcessScene to prevent
    120         // recursive calls when recomputing transients...
    121         fNeedKernelVisit = false;
    122         fSceneHandler.ProcessScene (*this);
    123       }
    124     }
    125

    (gdb) frame 11
    #11 0xb709a9f2 in G4VSceneHandler::ProcessScene (this=0x1024ab10) at src/G4VSceneHandler.cc:516
    516           pModel -> DescribeYourselfTo (*this);
    (gdb) list
    511           // pModel->GetTransformation().  The model must take care of
    512           // this in pModel->DescribeYourselfTo(*this).  See, for example,
    513           // G4PhysicalVolumeModel and /vis/scene/add/logo.
    514           pModel -> SetModelingParameters (pMP);
    515           SetModel (pModel);  // Store for use by derived class.
    516           pModel -> DescribeYourselfTo (*this);
    517           pModel -> SetModelingParameters (0);
    518         }
    519
    520         // Repeat if required...

    (gdb) frame 10
    #10 0xb5ef8219 in G4PhysicalVolumeModel::DescribeYourselfTo (this=0x1024b0c0, sceneHandler=@0x1024ab10) at src/G4PhysicalVolumeModel.cc:170
    170          sceneHandler);
    (gdb) list
    165
    166       VisitGeometryAndGetVisReps
    167         (fpTopPV,
    168          fRequestedDepth,
    169          startingTransformation,
    170          sceneHandler);
    171

    (gdb) frame 9 
    #9  0xb5ef6ba3 in G4PhysicalVolumeModel::VisitGeometryAndGetVisReps (this=0x1024b0c0, pVPV=0xbc0f528, requestedDepth=-1, theAT=@0xbfabbf80, sceneHandler=@0x1024ab10) at src/G4PhysicalVolumeModel.cc:222
    222                             theAT, sceneHandler);
    (gdb) list
    217       if (!(pVPV -> IsReplicated ())) {
    218         // Non-replicated physical volume.
    219         pSol = pLV -> GetSolid ();
    220         pMaterial = pLV -> GetMaterial ();
    221         DescribeAndDescend (pVPV, requestedDepth, pLV, pSol, pMaterial,
    222                             theAT, sceneHandler);
    223       }
    224       else {
    225         // Replicated or parametrised physical volume.
    226         EAxis axis;




Traversing scene data
~~~~~~~~~~~~~~~~~~~~~~

::

    [blyth@cms01 source]$ find . -name '*.cc' -exec grep -H Traversing {} \;
    ./visualization/management/src/G4VSceneHandler.cc:      G4cout << "Traversing scene data..." << G4endl;

::

    471 void G4VSceneHandler::ProcessScene (G4VViewer&) {
    472 
    473   if (!fpScene) return;
    474 
    475   G4VisManager* visManager = G4VisManager::GetInstance();
    476 
    477   if (!visManager->GetConcreteInstance()) return;
    478 
    479   G4VisManager::Verbosity verbosity = visManager->GetVerbosity();
    480 
    481   fReadyForTransients = false;
    482 
    483   // Clear stored scene, if any, i.e., display lists, scene graphs.
    484   ClearStore ();
    485 
    486   // Reset fMarkForClearingTransientStore.  No need to clear transient
    487   // store since it has just been cleared above.  (Leaving
    488   // fMarkForClearingTransientStore true causes problems with
    489   // recomputing transients below.)  Restore it again at end...
    490   G4bool tmpMarkForClearingTransientStore = fMarkForClearingTransientStore;
    491   fMarkForClearingTransientStore = false;
    492 
    493   // Traverse geometry tree and send drawing primitives to window(s).
    494 
    495   const std::vector<G4VModel*>& runDurationModelList =
    496     fpScene -> GetRunDurationModelList ();
    497 
    498   if (runDurationModelList.size ()) {
    499     if (verbosity >= G4VisManager::confirmations) {
    500       G4cout << "Traversing scene data..." << G4endl;
    501     }
    502 
    503     BeginModeling ();
    504 
    505     // Create modeling parameters from view parameters...
    506     G4ModelingParameters* pMP = CreateModelingParameters ();
    507 
    508     for (size_t i = 0; i < runDurationModelList.size (); i++) {
    509       G4VModel* pModel = runDurationModelList[i];
    510       // Note: this is not the place to take action on
    511       // pModel->GetTransformation().  The model must take care of
    512       // this in pModel->DescribeYourselfTo(*this).  See, for example,
    513       // G4PhysicalVolumeModel and /vis/scene/add/logo.
    514       pModel -> SetModelingParameters (pMP);
    515       SetModel (pModel);  // Store for use by derived class.
    516       pModel -> DescribeYourselfTo (*this);
    517       pModel -> SetModelingParameters (0);
    518     }
    519 
    520     // Repeat if required...
    521     if (fSecondPassRequested) {




runDurationModelList
~~~~~~~~~~~~~~~~~~~~~~~

$DYB/external/build/LCG/geant4.9.2.p01/source/visualization/management/include/G4Scene.hh::

     74   const std::vector<G4VModel*>& GetRunDurationModelList () const;
     75   // Contains models which are expected to last for the duration of
     76   // the run, for example geometry volumes.


The scene and this list are populated via draw commands.

$DYB/external/build/LCG/geant4.9.2.p01/source/visualization/management/src/G4VisCommandsCompound.cc::

    170 ////////////// /vis/drawVolume ///////////////////////////////////////
    171 
    172 G4VisCommandDrawVolume::G4VisCommandDrawVolume() {
    173   G4bool omitable;
    174   fpCommand = new G4UIcmdWithAString("/vis/drawVolume", this);
    175   fpCommand->SetGuidance
    176     ("Creates a scene containing this physical volume and asks the"
    177      "\ncurrent viewer to draw it.  The scene becomes current.");
    178   fpCommand -> SetGuidance
    179     ("If physical-volume-name is \"world\" (the default), the main geometry"
    180      "\ntree (material world) is drawn.  If \"worlds\", all worlds - material"
    181      "\nworld and parallel worlds, if any - are drawn.  Otherwise a search of"
    182      "\nall worlds is made, taking the first matching occurence only.  To see"
    183      "\na representation of the geometry hierarchy of the worlds, try"
    184      "\n\"/vis/drawTree [worlds]\" or one of the driver/browser combinations"
    185      "\nthat have the required functionality, e.g., HepRep");
    186   fpCommand->SetParameterName("physical-volume-name", omitable = true);
    187   fpCommand->SetDefaultValue("world");
    188 }

$DYB/external/build/LCG/geant4.9.2.p01/source/intercoms/src/G4UIcommand.cc::

    104 G4int G4UIcommand::DoIt(G4String parameterList)
    105 {
    210   messenger->SetNewValue( this, correctParameters );
    211   return 0;
    212 }

Invoking/Applying command boils down to a SetNewValue which applys `/vis/scene/add/volume World`::

    194 void G4VisCommandDrawVolume::SetNewValue(G4UIcommand*, G4String newValue) {
    195   G4VisManager::Verbosity verbosity = fpVisManager->GetVerbosity();
    196   G4UImanager* UImanager = G4UImanager::GetUIpointer();
    197   G4int keepVerbose = UImanager->GetVerboseLevel();
    198   G4int newVerbose(0);
    199   if (keepVerbose >= 2 || verbosity >= G4VisManager::confirmations)
    200     newVerbose = 2;
    201   UImanager->SetVerboseLevel(newVerbose);
    202   UImanager->ApplyCommand("/vis/scene/create");
    203   UImanager->ApplyCommand(G4String("/vis/scene/add/volume " + newValue));
    204   UImanager->ApplyCommand("/vis/sceneHandler/attach");
    205   UImanager->SetVerboseLevel(keepVerbose);
    206   static G4bool warned = false;
    207   if (verbosity >= G4VisManager::confirmations && !warned) {
    208     G4cout <<
    209       "NOTE: For systems which are not \"auto-refresh\" you will need to"
    210       "\n  issue \"/vis/viewer/refresh\" or \"/vis/viewer/flush\"."
    211        << G4endl;
    212     warned = true;
    213   }
    214 }


$DYB/external/build/LCG/geant4.9.2.p01/source/visualization/management/src/G4VisCommandsSceneAdd.cc::

   1479 ////////////// /vis/scene/add/volume ///////////////////////////////////////
   1480 
   1481 G4VisCommandSceneAddVolume::G4VisCommandSceneAddVolume () {
   1482   G4bool omitable;
   1483   fpCommand = new G4UIcommand ("/vis/scene/add/volume", this);
   1484   fpCommand -> SetGuidance
   1485    ("Adds a physical volume to current scene, with optional clipping volume.");
   1486   fpCommand -> SetGuidance
   1487     ("If physical-volume-name is \"world\" (the default), the top of the"
   1488      "\nmain geometry tree (material world) is added.  If \"worlds\", the"
   1489      "\ntop of all worlds - material world and parallel worlds, if any - are"
   1490      "\nadded.  Otherwise a search of all worlds is made, taking the first"
   1491      "\nmatching occurence only.  To see a representation of the geometry"
   1492      "\nhierarchy of the worlds, try \"/vis/drawTree [worlds]\" or one of the"
   1493      "\ndriver/browser combinations that have the required functionality, e.g., HepRep.");
   ....
   1558 void G4VisCommandSceneAddVolume::SetNewValue (G4UIcommand*,
   1559                           G4String newValue) {
   1560 
   1561   G4VisManager::Verbosity verbosity = fpVisManager->GetVerbosity();
   1562   G4bool warn = verbosity >= G4VisManager::warnings;
   1563 
   1564   G4Scene* pScene = fpVisManager->GetCurrentScene();
   ....
   1654   std::vector<G4PhysicalVolumeModel*> models;
   1655   std::vector<G4VPhysicalVolume*> foundVolumes;
   1656   G4VPhysicalVolume* foundWorld = 0;
   1657   std::vector<G4int> foundDepths;
   1658   std::vector<G4Transform3D> transformations;
   1659 
   1660   if (name == "world") {
   1661 
   1662     models.push_back
   1663       (new G4PhysicalVolumeModel (world, requestedDepthOfDescent));
   1664     foundVolumes.push_back(world);
   1665     foundDepths.push_back(0);
   1666     transformations.push_back(G4Transform3D());
   ....
   1746   const G4String& currentSceneName = pScene -> GetName ();
   1747   G4bool failure = true;
   1748   for (size_t i = 0; i < foundVolumes.size(); ++i) {
   1749     G4bool successful = pScene -> AddRunDurationModel (models[i], warn);
   1750     if (successful) {


$DYB/external/build/LCG/geant4.9.2.p01/source/visualization/management/include/G4Scene.hh::

    108   G4bool AddRunDurationModel (G4VModel*, G4bool warn = false);
    109   // Adds models of type which are expected to last for the duration
    110   // of the run, for example geometry volumes.
    111   // Returns false if model is already in the list.
    112   // Prints warnings if warn is true.
    113 


$DYB/external/build/LCG/geant4.9.2.p01/source/visualization/management/src/G4Scene.cc::

     50 G4bool G4Scene::AddRunDurationModel (G4VModel* pModel, G4bool warn) {
     51   std::vector<G4VModel*>::const_iterator i;
     52   for (i = fRunDurationModelList.begin ();
     53        i != fRunDurationModelList.end (); ++i) {
     54     if (pModel -> GetGlobalDescription () ==
     55     (*i) -> GetGlobalDescription ()) break;
     56   }
     57   if (i != fRunDurationModelList.end ()) {
     58     if (warn) {
     59       G4cout << "G4Scene::AddRunDurationModel: model \""
     60          << pModel -> GetGlobalDescription ()
     61          << "\"\n  is already in the run-duration list of scene \""
     62          << fName
     63          << "\"."
     64          << G4endl;
     65     }
     66     return false;
     67   }
     68   fRunDurationModelList.push_back (pModel);
     69   CalculateExtent ();
     70   return true;
     71 }


According to my reading the RunDurationModel contains just world. Actually `G4PhysicalVolumeModel (world, requestedDepthOfDescent))`


G4PhysicalVolumeModel
-----------------------


$DYB/external/build/LCG/geant4.9.2.p01/source/visualization/modeling/include/G4PhysicalVolumeModel.hh::

     67 class G4PhysicalVolumeModel: public G4VModel {
     68 
     69 public: // With description
     70 
     71   enum {UNLIMITED = -1};
     72 
     73   enum ClippingMode {subtraction, intersection};
     74 
     75   class G4PhysicalVolumeNodeID {
     76   public:
     77     G4PhysicalVolumeNodeID
     78     (G4VPhysicalVolume* pPV = 0, G4int iCopyNo = 0, G4int depth = 0):
     79       fpPV(pPV), fCopyNo(iCopyNo), fNonCulledDepth(depth) {}
     80     G4VPhysicalVolume* GetPhysicalVolume() const {return fpPV;}
     81     G4int GetCopyNo() const {return fCopyNo;}
     82     G4int GetNonCulledDepth() const {return fNonCulledDepth;}
     83     G4bool operator< (const G4PhysicalVolumeNodeID& right) const;
     84   private:
     85     G4VPhysicalVolume* fpPV;
     86     G4int fCopyNo;
     87     G4int fNonCulledDepth;
     88   };
     89   // Nested class for identifying physical volume nodes.
     90 
     91   G4PhysicalVolumeModel
     92   (G4VPhysicalVolume*,
     93    G4int requestedDepth = UNLIMITED,
     94    const G4Transform3D& modelTransformation = G4Transform3D(),
     95    const G4ModelingParameters* = 0,
     96    G4bool useFullExtent = false);
     97 
     98   virtual ~G4PhysicalVolumeModel ();
     99 
     00   void DescribeYourselfTo (G4VGraphicsScene&);
     01   // The main task of a model is to describe itself to the graphics scene
     02   // handler (a object which inherits G4VSceneHandler, which inherits
     03   // G4VGraphicsScene).

$DYB/external/build/LCG/geant4.9.2.p01/source/visualization/modeling/src/G4PhysicalVolumeModel.cc::

    /// kick off the descent
    ///
    155 void G4PhysicalVolumeModel::DescribeYourselfTo
    156 (G4VGraphicsScene& sceneHandler)
    157 {
    158   if (!fpMP) G4Exception
    159     ("G4PhysicalVolumeModel::DescribeYourselfTo: No modeling parameters.");
    160 
    161   // For safety...
    162   fCurrentDepth = 0;
    163 
    164   G4Transform3D startingTransformation = fTransform;
    165 
    166   VisitGeometryAndGetVisReps
    167     (fpTopPV,
    168      fRequestedDepth,
    169      startingTransformation,
    170      sceneHandler);
    ///
    ///   handle the recursive descent 
    ///
    198 void G4PhysicalVolumeModel::VisitGeometryAndGetVisReps
    199 (G4VPhysicalVolume* pVPV,
    200  G4int requestedDepth,
    201  const G4Transform3D& theAT,
    202  G4VGraphicsScene& sceneHandler)
    203 {
    204   // Visits geometry structure to a given depth (requestedDepth), starting
    205   //   at given physical volume with given starting transformation and
    206   //   describes volumes to the scene handler.
    207   // requestedDepth < 0 (default) implies full visit.
    208   // theAT is the Accumulated Transformation.
    209 
    210   // Find corresponding logical volume and (later) solid, storing in
    211   // local variables to preserve re-entrancy.
    212   G4LogicalVolume* pLV  = pVPV -> GetLogicalVolume ();
    213 
    214   G4VSolid* pSol;
    215   G4Material* pMaterial;
    216 
    217   if (!(pVPV -> IsReplicated ())) {
    218     // Non-replicated physical volume.
    219     pSol = pLV -> GetSolid ();
    220     pMaterial = pLV -> GetMaterial ();
    221     DescribeAndDescend (pVPV, requestedDepth, pLV, pSol, pMaterial,
    222             theAT, sceneHandler);
    ///
    ///
    ///
    336 void G4PhysicalVolumeModel::DescribeAndDescend
    337 (G4VPhysicalVolume* pVPV,
    338  G4int requestedDepth,
    339  G4LogicalVolume* pLV,
    340  G4VSolid* pSol,
    341  G4Material* pMaterial,
    342  const G4Transform3D& theAT,
    343  G4VGraphicsScene& sceneHandler)
    344 {
    345   // Maintain useful data members...
    346   fpCurrentPV = pVPV;
    347   fpCurrentLV = pLV;
    348   fpCurrentMaterial = pMaterial;
    349 
    350   const G4RotationMatrix objectRotation = pVPV -> GetObjectRotationValue ();
    351   const G4ThreeVector&  translation     = pVPV -> GetTranslation ();
    352   G4Transform3D theLT (G4Transform3D (objectRotation, translation));
    353 
    354   // Compute the accumulated transformation...
    355   // Note that top volume's transformation relative to the world
    356   // coordinate system is specified in theAT == startingTransformation
    357   // = fTransform (see DescribeYourselfTo), so first time through the
    358   // volume's own transformation, which is only relative to its
    359   // mother, i.e., not relative to the world coordinate system, should
    360   // not be accumulated.
    361   G4Transform3D theNewAT (theAT);
    362   if (fCurrentDepth != 0) theNewAT = theAT * theLT;
    363   fpCurrentTransform = &theNewAT;
    364 
    ...
    428   // Update full path of physical volumes...
    429   G4int copyNo = fpCurrentPV->GetCopyNo();
    430   fFullPVPath.push_back
    431     (G4PhysicalVolumeNodeID(fpCurrentPV,copyNo,fCurrentDepth));
    ...
    433   if (thisToBeDrawn) {
    434 
    435     // Update path of drawn physical volumes...
    436     G4int copyNo = fpCurrentPV->GetCopyNo();
    437     fDrawnPVPath.push_back
    438       (G4PhysicalVolumeNodeID(fpCurrentPV,copyNo,fCurrentDepth));
    439 
    440     if (fpMP->IsExplode() && fDrawnPVPath.size() == 1) {
    ...
    454     }
    455 
    456     DescribeSolid (theNewAT, pSol, pVisAttribs, sceneHandler);
    457 
    458   }
    ...
    517   if (daughtersToBeDrawn) {
    518     for (G4int iDaughter = 0; iDaughter < nDaughters; iDaughter++) {
    519       G4VPhysicalVolume* pVPV = pLV -> GetDaughter (iDaughter);
    520       // Descend the geometry structure recursively...
    521       fCurrentDepth++;
    522       VisitGeometryAndGetVisReps
    523     (pVPV, requestedDepth - 1, theNewAT, sceneHandler);
    524       fCurrentDepth--;
    525     }
    526   }
    ...
    538 void G4PhysicalVolumeModel::DescribeSolid
    539 (const G4Transform3D& theAT,
    540  G4VSolid* pSol,
    541  const G4VisAttributes* pVisAttribs,
    542  G4VGraphicsScene& sceneHandler)
    543 {
    544   sceneHandler.PreAddSolid (theAT, *pVisAttribs);
    545 
    546   const G4Polyhedron* pSectionPolyhedron = fpMP->GetSectionPolyhedron();
    547   const G4Polyhedron* pCutawayPolyhedron = fpMP->GetCutawayPolyhedron();
    548 
    549   if (!fpClippingPolyhedron && !pSectionPolyhedron && !pCutawayPolyhedron) {
    550 
    551     pSol -> DescribeYourselfTo (sceneHandler);  // Standard treatment.
    552 
    553   } else {
    554 
    555     // Clipping, etc., performed by Boolean operations on polyhedron objects.
    ...
    633   sceneHandler.PostAddSolid ();
    634 }




DescribeYourselfTo
--------------------

::

    [blyth@cms01 src]$ find $DYB/external/build/LCG/geant4.9.2.p01/source -name '*.cc' -exec grep -H DescribeYourselfTo {} \;
    /data/env/local/dyb/trunk/external/build/LCG/geant4.9.2.p01/source/visualization/management/src/G4VisCommandsSceneAdd.cc:      searchModel.DescribeYourselfTo (searchScene);  // Initiate search.
    /data/env/local/dyb/trunk/external/build/LCG/geant4.9.2.p01/source/visualization/management/src/G4VSceneHandler.cc:      // this in pModel->DescribeYourselfTo(*this).  See, for example,
    /data/env/local/dyb/trunk/external/build/LCG/geant4.9.2.p01/source/visualization/management/src/G4VSceneHandler.cc:      pModel -> DescribeYourselfTo (*this);
    /data/env/local/dyb/trunk/external/build/LCG/geant4.9.2.p01/source/visualization/management/src/G4VSceneHandler.cc:     pModel -> DescribeYourselfTo (*this);
    /data/env/local/dyb/trunk/external/build/LCG/geant4.9.2.p01/source/visualization/management/src/G4VSceneHandler.cc:      pModel -> DescribeYourselfTo (*this);
    /data/env/local/dyb/trunk/external/build/LCG/geant4.9.2.p01/source/visualization/management/src/G4VisManager.cc:    solid.DescribeYourselfTo (*fpSceneHandler);
    /data/env/local/dyb/trunk/external/build/LCG/geant4.9.2.p01/source/visualization/modeling/src/G4AxesModel.cc:void G4AxesModel::DescribeYourselfTo (G4VGraphicsScene& sceneHandler) {
    /data/env/local/dyb/trunk/external/build/LCG/geant4.9.2.p01/source/visualization/modeling/src/G4PhysicalVolumeModel.cc:    DescribeYourselfTo (bsScene);
    /data/env/local/dyb/trunk/external/build/LCG/geant4.9.2.p01/source/visualization/modeling/src/G4PhysicalVolumeModel.cc:void G4PhysicalVolumeModel::DescribeYourselfTo
    /data/env/local/dyb/trunk/external/build/LCG/geant4.9.2.p01/source/visualization/modeling/src/G4PhysicalVolumeModel.cc:    ("G4PhysicalVolumeModel::DescribeYourselfTo: No modeling parameters.");
    /data/env/local/dyb/trunk/external/build/LCG/geant4.9.2.p01/source/visualization/modeling/src/G4PhysicalVolumeModel.cc:  // = fTransform (see DescribeYourselfTo), so first time through the
    /data/env/local/dyb/trunk/external/build/LCG/geant4.9.2.p01/source/visualization/modeling/src/G4PhysicalVolumeModel.cc:    pSol -> DescribeYourselfTo (sceneHandler);  // Standard treatment.
    /data/env/local/dyb/trunk/external/build/LCG/geant4.9.2.p01/source/visualization/modeling/src/G4PhysicalVolumeModel.cc: pSol -> DescribeYourselfTo (sceneHandler);  // Standard treatment.
    /data/env/local/dyb/trunk/external/build/LCG/geant4.9.2.p01/source/visualization/modeling/src/G4PhysicalVolumeModel.cc:  searchModel.DescribeYourselfTo (searchScene);
    /data/env/local/dyb/trunk/external/build/LCG/geant4.9.2.p01/source/visualization/modeling/src/G4HitsModel.cc:void G4HitsModel::DescribeYourselfTo (G4VGraphicsScene& sceneHandler)
    /data/env/local/dyb/trunk/external/build/LCG/geant4.9.2.p01/source/visualization/modeling/src/G4ScaleModel.cc:void G4ScaleModel::DescribeYourselfTo (G4VGraphicsScene& sceneHandler) {
    /data/env/local/dyb/trunk/external/build/LCG/geant4.9.2.p01/source/visualization/modeling/src/G4TrajectoriesModel.cc:void G4TrajectoriesModel::DescribeYourselfTo (G4VGraphicsScene& sceneHandler) {
    /data/env/local/dyb/trunk/external/build/LCG/geant4.9.2.p01/source/visualization/modeling/src/G4LogicalVolumeModel.cc:void G4LogicalVolumeModel::DescribeYourselfTo
    /data/env/local/dyb/trunk/external/build/LCG/geant4.9.2.p01/source/visualization/modeling/src/G4LogicalVolumeModel.cc:  G4PhysicalVolumeModel::DescribeYourselfTo (sceneHandler);
    /data/env/local/dyb/trunk/external/build/LCG/geant4.9.2.p01/source/visualization/modeling/src/G4LogicalVolumeModel.cc:  pvModel.DescribeYourselfTo(sceneHandler);
    /data/env/local/dyb/trunk/external/build/LCG/geant4.9.2.p01/source/visualization/modeling/src/G4LogicalVolumeModel.cc:  pSol -> DescribeYourselfTo (sceneHandler);
    /data/env/local/dyb/trunk/external/build/LCG/geant4.9.2.p01/source/visualization/modeling/src/G4NullModel.cc:void G4NullModel::DescribeYourselfTo (G4VGraphicsScene&) {
    /data/env/local/dyb/trunk/external/build/LCG/geant4.9.2.p01/source/visualization/modeling/src/G4NullModel.cc:  G4Exception ("G4NullModel::DescribeYourselfTo called.");
    /data/env/local/dyb/trunk/external/build/LCG/geant4.9.2.p01/source/visualization/modeling/src/G4TextModel.cc:void G4TextModel::DescribeYourselfTo (G4VGraphicsScene& sceneHandler) {
    /data/env/local/dyb/trunk/external/build/LCG/geant4.9.2.p01/source/visualization/Tree/src/G4ASCIITreeSceneHandler.cc:     pvModel->DescribeYourselfTo (massScene);
    /data/env/local/dyb/trunk/external/build/LCG/geant4.9.2.p01/source/geometry/management/src/G4ReflectedSolid.cc:G4ReflectedSolid::DescribeYourselfTo ( G4VGraphicsScene& scene ) const 
    /data/env/local/dyb/trunk/external/build/LCG/geant4.9.2.p01/source/geometry/solids/BREPS/src/G4BREPSolid.cc:void G4BREPSolid::DescribeYourselfTo (G4VGraphicsScene& scene) const 
    /data/env/local/dyb/trunk/external/build/LCG/geant4.9.2.p01/source/geometry/solids/BREPS/src/G4BREPSolidOpenPCone.cc:void G4BREPSolidOpenPCone::DescribeYourselfTo (G4VGraphicsScene& scene) const
    /data/env/local/dyb/trunk/external/build/LCG/geant4.9.2.p01/source/geometry/solids/CSG/src/G4Tubs.cc:void G4Tubs::DescribeYourselfTo ( G4VGraphicsScene& scene ) const 
    /data/env/local/dyb/trunk/external/build/LCG/geant4.9.2.p01/source/geometry/solids/CSG/src/G4Torus.cc:void G4Torus::DescribeYourselfTo ( G4VGraphicsScene& scene ) const 
    /data/env/local/dyb/trunk/external/build/LCG/geant4.9.2.p01/source/geometry/solids/CSG/src/G4Trap.cc:void G4Trap::DescribeYourselfTo ( G4VGraphicsScene& scene ) const
    /data/env/local/dyb/trunk/external/build/LCG/geant4.9.2.p01/source/geometry/solids/CSG/src/G4Para.cc:void G4Para::DescribeYourselfTo ( G4VGraphicsScene& scene ) const
    /data/env/local/dyb/trunk/external/build/LCG/geant4.9.2.p01/source/geometry/solids/CSG/src/G4Cons.cc:void G4Cons::DescribeYourselfTo (G4VGraphicsScene& scene) const
    /data/env/local/dyb/trunk/external/build/LCG/geant4.9.2.p01/source/geometry/solids/CSG/src/G4Trd.cc:void G4Trd::DescribeYourselfTo ( G4VGraphicsScene& scene ) const
    /data/env/local/dyb/trunk/external/build/LCG/geant4.9.2.p01/source/geometry/solids/CSG/src/G4Sphere.cc:void G4Sphere::DescribeYourselfTo ( G4VGraphicsScene& scene ) const
    /data/env/local/dyb/trunk/external/build/LCG/geant4.9.2.p01/source/geometry/solids/CSG/src/G4Orb.cc:void G4Orb::DescribeYourselfTo ( G4VGraphicsScene& scene ) const
    /data/env/local/dyb/trunk/external/build/LCG/geant4.9.2.p01/source/geometry/solids/CSG/src/G4Box.cc:void G4Box::DescribeYourselfTo (G4VGraphicsScene& scene) const 
    /data/env/local/dyb/trunk/external/build/LCG/geant4.9.2.p01/source/geometry/solids/specific/src/G4VCSGfaceted.cc:// DescribeYourselfTo
    /data/env/local/dyb/trunk/external/build/LCG/geant4.9.2.p01/source/geometry/solids/specific/src/G4VCSGfaceted.cc:void G4VCSGfaceted::DescribeYourselfTo( G4VGraphicsScene& scene ) const
    /data/env/local/dyb/trunk/external/build/LCG/geant4.9.2.p01/source/geometry/solids/specific/src/G4Tet.cc:// DescribeYourselfTo
    /data/env/local/dyb/trunk/external/build/LCG/geant4.9.2.p01/source/geometry/solids/specific/src/G4Tet.cc:void G4Tet::DescribeYourselfTo (G4VGraphicsScene& scene) const 
    /data/env/local/dyb/trunk/external/build/LCG/geant4.9.2.p01/source/geometry/solids/specific/src/G4EllipticalTube.cc:// DescribeYourselfTo
    /data/env/local/dyb/trunk/external/build/LCG/geant4.9.2.p01/source/geometry/solids/specific/src/G4EllipticalTube.cc:void G4EllipticalTube::DescribeYourselfTo( G4VGraphicsScene& scene ) const
    /data/env/local/dyb/trunk/external/build/LCG/geant4.9.2.p01/source/geometry/solids/specific/src/G4Hype.cc:// DescribeYourselfTo
    /data/env/local/dyb/trunk/external/build/LCG/geant4.9.2.p01/source/geometry/solids/specific/src/G4Hype.cc:void G4Hype::DescribeYourselfTo (G4VGraphicsScene& scene) const 
    /data/env/local/dyb/trunk/external/build/LCG/geant4.9.2.p01/source/geometry/solids/specific/src/G4VTwistedFaceted.cc:void G4VTwistedFaceted::DescribeYourselfTo (G4VGraphicsScene& scene) const 
    /data/env/local/dyb/trunk/external/build/LCG/geant4.9.2.p01/source/geometry/solids/specific/src/G4TessellatedSolid.cc://  - Changed ::DescribeYourselfTo(), line 464
    /data/env/local/dyb/trunk/external/build/LCG/geant4.9.2.p01/source/geometry/solids/specific/src/G4TessellatedSolid.cc:void G4TessellatedSolid::DescribeYourselfTo (G4VGraphicsScene& scene) const
    /data/env/local/dyb/trunk/external/build/LCG/geant4.9.2.p01/source/geometry/solids/specific/src/G4Paraboloid.cc:void G4Paraboloid::DescribeYourselfTo (G4VGraphicsScene& scene) const
    /data/env/local/dyb/trunk/external/build/LCG/geant4.9.2.p01/source/geometry/solids/specific/src/G4Ellipsoid.cc:void G4Ellipsoid::DescribeYourselfTo (G4VGraphicsScene& scene) const
    /data/env/local/dyb/trunk/external/build/LCG/geant4.9.2.p01/source/geometry/solids/specific/src/G4EllipticalCone.cc:void G4EllipticalCone::DescribeYourselfTo (G4VGraphicsScene& scene) const
    /data/env/local/dyb/trunk/external/build/LCG/geant4.9.2.p01/source/geometry/solids/specific/src/G4TwistedTubs.cc:void G4TwistedTubs::DescribeYourselfTo (G4VGraphicsScene& scene) const 
    /data/env/local/dyb/trunk/external/build/LCG/geant4.9.2.p01/source/geometry/solids/Boolean/src/G4DisplacedSolid.cc:G4DisplacedSolid::DescribeYourselfTo ( G4VGraphicsScene& scene ) const 
    /data/env/local/dyb/trunk/external/build/LCG/geant4.9.2.p01/source/geometry/solids/Boolean/src/G4IntersectionSolid.cc:G4IntersectionSolid::DescribeYourselfTo ( G4VGraphicsScene& scene ) const 
    /data/env/local/dyb/trunk/external/build/LCG/geant4.9.2.p01/source/geometry/solids/Boolean/src/G4SubtractionSolid.cc:G4SubtractionSolid::DescribeYourselfTo ( G4VGraphicsScene& scene ) const 
    /data/env/local/dyb/trunk/external/build/LCG/geant4.9.2.p01/source/geometry/solids/Boolean/src/G4UnionSolid.cc:G4UnionSolid::DescribeYourselfTo ( G4VGraphicsScene& scene ) const 
    [blyth@cms01 src]$ 


::

   1885 void G4Tubs::DescribeYourselfTo ( G4VGraphicsScene& scene ) const
   1886 {
   1887   scene.AddSolid (*this) ;
   1888 }


::

    443 void
    444 G4UnionSolid::DescribeYourselfTo ( G4VGraphicsScene& scene ) const
    445 {
    446   scene.AddSolid (*this);
    447 }



$DYB/external/build/LCG/geant4.9.2.p01/source/visualization/modeling/src/G4LogicalVolumeModel.cc::

     //
     //  POSSIBLE EXPLANATION FOR GETTING MORE VRML2 SOLIDS THAN EXPECTED IN THE .wrl ???  
     //  THE LOGICALS ARE BEING DRAWN 
     //
     //
     45 G4LogicalVolumeModel::G4LogicalVolumeModel
     46 (G4LogicalVolume*            pLV,
     47  G4int                       soughtDepth,
     48  G4bool                      booleans,
     49  G4bool                      voxels,
     50  G4bool                      readout,
     51  const G4Transform3D&        modelTransformation,
     52  const G4ModelingParameters* pMP):
     53   // Instantiate a G4PhysicalVolumeModel with a G4PVPlacement to
     54   // represent this logical volume.  It has no rotation and a null
     55   // translation so that the logical volume will be seen in its own
     56   // reference system.  It will be added to the physical volume store
     57   // but it will not be part of the normal geometry heirarchy so it
     58   // has no mother.
     59   G4PhysicalVolumeModel
     60 (new G4PVPlacement (0,                   // No rotation.
     61             G4ThreeVector(),     // Null traslation.
     62             "PhysVol representaion of LogVol " + pLV -> GetName (),
     63             pLV,
     64             0,                   // No mother.
     65             false,               // Not "MANY".
     66             0),                  // Copy number.
     67  soughtDepth,
     ..
     82 void G4LogicalVolumeModel::DescribeYourselfTo
     83 (G4VGraphicsScene& sceneHandler) {
     84 
     85   // Store current modeling parameters and ensure nothing is culled.
     86   const G4ModelingParameters* tmpMP = fpMP;
     87   G4ModelingParameters nonCulledMP;
     88   if (fpMP) nonCulledMP = *fpMP;
     89   nonCulledMP.SetCulling (false);
     90   fpMP = &nonCulledMP;
     91   G4PhysicalVolumeModel::DescribeYourselfTo (sceneHandler);
     92   fpMP = tmpMP;


::
    
    130 // This called from G4PhysicalVolumeModel::DescribeAndDescend by the
    131 // virtual function mechanism.
    132 void G4LogicalVolumeModel::DescribeSolid
    133 (const G4Transform3D& theAT,
    134  G4VSolid* pSol,
    135  const G4VisAttributes* pVisAttribs,
    136  G4VGraphicsScene& sceneHandler) {
    137 
    138   if (fBooleans) {
    139     // Look for "constituents".  Could be a Boolean solid.
    140     G4VSolid* pSol0 = pSol -> GetConstituentSolid (0);
    141     if (pSol0) {  // Composite solid...
    142       G4VSolid* pSol1 = pSol -> GetConstituentSolid (1);
    143       if (!pSol1) {
    144     G4Exception
    145       ("G4PhysicalVolumeModel::DescribeSolid:"
    146        " 2nd component solid is missing.");
    147       }
    148       // Draw these constituents white and "forced wireframe"...
    149       G4VisAttributes constituentAttributes;
    150       constituentAttributes.SetForceWireframe(true);
    151       DescribeSolid (theAT, pSol0, &constituentAttributes, sceneHandler);
    152       DescribeSolid (theAT, pSol1, &constituentAttributes, sceneHandler);
    153     }
    154   }
    155 
    156   // In any case draw the original/resultant solid...
    157   sceneHandler.PreAddSolid (theAT, *pVisAttribs);
    158   pSol -> DescribeYourselfTo (sceneHandler);
    159   sceneHandler.PostAddSolid ();
    160 }



$DYB/external/build/LCG/geant4.9.2.p01/source/visualization/VRML/src/G4VRML2SceneHandlerFunc.icc::

    111 void G4VRML2SCENEHANDLER::AddSolid(const G4Sphere& sphere)
    112 {
    113 #if defined DEBUG_SCENE_FUNC
    114     G4cerr << "***** AddSolid sphere" << "\n" ;
    115 #endif
    116     VRMLBeginModeling () ;
    117     G4VSceneHandler::AddSolid(sphere) ;
    118 }
    ...
    ...
    169 void G4VRML2SCENEHANDLER::AddPrimitive(const G4Polyhedron& polyhedron)
    170 {
    ...
    174 
    175     if (polyhedron.GetNoFacets() == 0) return;
    176 
    177     VRMLBeginModeling () ;
    178 
    179     // Transparency checking: If completely transparent, skip drawing
    180     if ( GetPVTransparency() > 0.99 ) { return ; }
    181 
    182     // Current Model
    183     const G4VModel* pv_model  = GetModel();
    184     G4String pv_name = "No model";
    185         if (pv_model) pv_name = pv_model->GetCurrentTag() ;
    186 
    187     // VRML codes are generated below
    188 
    189     fDest << "#---------- SOLID: " << pv_name << "\n";
    190 
    ...
    198 
    199     fDest << "\t"; fDest << "Shape {" << "\n";
    200 
    201     SendMaterialNode();
    202 
    203     fDest << "\t\t" << "geometry IndexedFaceSet {" << "\n";
    204 
    205     fDest << "\t\t\t"   << "coord Coordinate {" << "\n";
    206     fDest << "\t\t\t\t" <<      "point [" << "\n";
    207     G4int i, j;
    208     for (i = 1, j = polyhedron.GetNoVertices(); j; j--, i++) {
    209         G4Point3D point = polyhedron.GetVertex(i);
    210


$DYB/external/build/LCG/geant4.9.2.p01/source/visualization/management/include/G4VSceneHandler.hh::

    200 void G4VSceneHandler::AddSolid (const G4Box& box) {
    201   RequestPrimitives (box);
    202 // If your graphics system is sophisticated enough to handle a
    203 //  particular solid shape as a primitive, in your derived class write a
    204 //  function to override this.  (Note: some compilers warn that your
    205 //  function "hides" this one.  That's OK.)
    206 // Your function might look like this...
    207 // void G4MyScene::AddSolid (const G4Box& box) {
    208 // Get parameters of appropriate object, e.g.:
    209 //   G4double dx = box.GetXHalfLength ();
    210 //   G4double dy = box.GetYHalfLength ();
    211 //   G4double dz = box.GetZHalfLength ();
    212 // and Draw or Store in your display List.
    213 }
    214
    215 void G4VSceneHandler::AddSolid (const G4Tubs& tubs) {
    216   RequestPrimitives (tubs);
    217 }
    218 
    219 void G4VSceneHandler::AddSolid (const G4Cons& cons) {
    220   RequestPrimitives (cons);
    221 }
    ///
    ///      HERE THE POLYHEDRON REPRESENTATION OF THE SOLID SPRINGS INTO EXISTANCE
    ///
    421 void G4VSceneHandler::RequestPrimitives (const G4VSolid& solid) {
    422   BeginPrimitives (*fpObjectTransformation);
    423   G4NURBS* pNURBS = 0;
    424   G4Polyhedron* pPolyhedron = 0;
    425   switch (fpViewer -> GetViewParameters () . GetRepStyle ()) {
    426   case G4ViewParameters::nurbs:
    427     pNURBS = solid.CreateNURBS ();
    428     if (pNURBS) {
    429       pNURBS -> SetVisAttributes (fpVisAttribs);
    430       AddPrimitive (*pNURBS);
    431       delete pNURBS;
    432       break;
    433     }
    434     else {
    435       G4VisManager::Verbosity verbosity =
    436     G4VisManager::GetInstance()->GetVerbosity();
    437       if (verbosity >= G4VisManager::errors) {
    438     G4cout <<
    439       "ERROR: G4VSceneHandler::RequestPrimitives"
    440       "\n  NURBS not available for "
    441            << solid.GetName () << G4endl;
    442     G4cout << "Trying polyhedron." << G4endl;
    443       }
    444     }
    445     // Dropping through to polyhedron...
    446   case G4ViewParameters::polyhedron:
    447   default:
    448     G4Polyhedron::SetNumberOfRotationSteps (GetNoOfSides (fpVisAttribs));
    449     pPolyhedron = solid.GetPolyhedron ();
    450     G4Polyhedron::ResetNumberOfRotationSteps ();
    451     if (pPolyhedron) {
    452       pPolyhedron -> SetVisAttributes (fpVisAttribs);
    453       AddPrimitive (*pPolyhedron);
    454     }
    455     else {
    456       G4VisManager::Verbosity verbosity =
    457     G4VisManager::GetInstance()->GetVerbosity();
    458       if (verbosity >= G4VisManager::errors) {
    459     G4cout <<
    460       "ERROR: G4VSceneHandler::RequestPrimitives"
    461       "\n  Polyhedron not available for " << solid.GetName () <<
    462       ".\n  This means it cannot be visualized on most systems."
    463       "\n  Contact the Visualization Coordinator." << G4endl;
    464       }
    465     }
    466     break;
    467   }
    468   EndPrimitives ();
    469 }


GetPolyhedron
---------------

::

    [blyth@cms01 source]$ find . -name '*.hh' -exec grep -H GetPolyhedron {} \;
    ./graphics_reps/include/G4PlacedPolyhedron.hh:  const G4Polyhedron&  GetPolyhedron () const {return fPolyhedron;}
    ./geometry/management/include/G4VSolid.hh:    virtual G4Polyhedron* GetPolyhedron () const;
    ./geometry/management/include/G4ReflectedSolid.hh:    G4Polyhedron* GetPolyhedron    () const;
    ./geometry/solids/BREPS/include/G4BREPSolid.hh:  virtual G4Polyhedron* GetPolyhedron () const;
    ./geometry/solids/CSG/include/G4CSGSolid.hh:    virtual G4Polyhedron* GetPolyhedron () const;
    ./geometry/solids/specific/include/G4TessellatedSolid.hh://  - Added GetPolyhedron()
    ./geometry/solids/specific/include/G4TessellatedSolid.hh:    virtual G4Polyhedron* GetPolyhedron    () const;
    ./geometry/solids/specific/include/G4VCSGfaceted.hh:    virtual G4Polyhedron* GetPolyhedron () const;
    ./geometry/solids/specific/include/G4TwistedTubs.hh:  G4Polyhedron   *GetPolyhedron      () const;
    ./geometry/solids/specific/include/G4Paraboloid.hh:    G4Polyhedron* GetPolyhedron () const;
    ./geometry/solids/specific/include/G4EllipticalTube.hh:    G4Polyhedron* GetPolyhedron () const;
    ./geometry/solids/specific/include/G4Ellipsoid.hh:    G4Polyhedron* GetPolyhedron () const;
    ./geometry/solids/specific/include/G4Hype.hh:  G4Polyhedron* GetPolyhedron      () const;
    ./geometry/solids/specific/include/G4Tet.hh:    G4Polyhedron* GetPolyhedron      () const;
    ./geometry/solids/specific/include/G4VTwistedFaceted.hh:  virtual G4Polyhedron   *GetPolyhedron      () const;
    ./geometry/solids/specific/include/G4EllipticalCone.hh:    G4Polyhedron* GetPolyhedron () const;
    ./geometry/solids/Boolean/include/G4BooleanSolid.hh:    virtual G4Polyhedron* GetPolyhedron () const;
    ./geometry/solids/Boolean/include/G4DisplacedSolid.hh:    virtual G4Polyhedron* GetPolyhedron () const;
    [blyth@cms01 source]$ 


::

    [blyth@cms01 source]$ find . -name '*.cc' -exec grep -H GetPolyhedron {} \;
    ./visualization/management/src/G4VSceneHandler.cc:    pPolyhedron = solid.GetPolyhedron ();
    ./visualization/modeling/src/G4PhysicalVolumeModel.cc:    G4Polyhedron* pOriginal = pSol->GetPolyhedron();
    ./visualization/modeling/src/G4PhysicalVolumeMassScene.cc:  G4Polyhedron* pPolyhedron = solid.GetPolyhedron();
    ./visualization/modeling/src/G4LogicalVolumeModel.cc:   const G4Polyhedron& polyhedron = (*pPPL)[i].GetPolyhedron ();
    ./geometry/management/src/G4ReflectedSolid.cc:G4ReflectedSolid::GetPolyhedron () const
    ./geometry/management/src/G4VSolid.cc:G4Polyhedron* G4VSolid::GetPolyhedron () const
    ./geometry/solids/BREPS/src/G4BREPSolid.cc:G4Polyhedron* G4BREPSolid::GetPolyhedron () const
    ./geometry/solids/CSG/src/G4CSGSolid.cc:G4Polyhedron* G4CSGSolid::GetPolyhedron () const
    ./geometry/solids/specific/src/G4VCSGfaceted.cc:// GetPolyhedron
    ./geometry/solids/specific/src/G4VCSGfaceted.cc:G4Polyhedron* G4VCSGfaceted::GetPolyhedron () const
    ./geometry/solids/specific/src/G4Tet.cc:// GetPolyhedron
    ./geometry/solids/specific/src/G4Tet.cc:G4Polyhedron* G4Tet::GetPolyhedron () const
    ./geometry/solids/specific/src/G4EllipticalTube.cc:// GetPolyhedron
    ./geometry/solids/specific/src/G4EllipticalTube.cc:G4Polyhedron* G4EllipticalTube::GetPolyhedron () const
    ./geometry/solids/specific/src/G4Hype.cc:// GetPolyhedron
    ./geometry/solids/specific/src/G4Hype.cc:G4Polyhedron* G4Hype::GetPolyhedron () const
    ./geometry/solids/specific/src/G4VTwistedFaceted.cc://* GetPolyhedron -----------------------------------------------------
    ./geometry/solids/specific/src/G4VTwistedFaceted.cc:G4Polyhedron* G4VTwistedFaceted::GetPolyhedron() const
    ./geometry/solids/specific/src/G4TessellatedSolid.cc:// GetPolyhedron
    ./geometry/solids/specific/src/G4TessellatedSolid.cc:G4Polyhedron* G4TessellatedSolid::GetPolyhedron () const
    ./geometry/solids/specific/src/G4Paraboloid.cc:G4Polyhedron* G4Paraboloid::GetPolyhedron () const
    ./geometry/solids/specific/src/G4Ellipsoid.cc:G4Polyhedron* G4Ellipsoid::GetPolyhedron () const
    ./geometry/solids/specific/src/G4EllipticalCone.cc:G4Polyhedron* G4EllipticalCone::GetPolyhedron () const
    ./geometry/solids/specific/src/G4TwistedTubs.cc://* GetPolyhedron -----------------------------------------------------
    ./geometry/solids/specific/src/G4TwistedTubs.cc:G4Polyhedron* G4TwistedTubs::GetPolyhedron () const
    ./geometry/solids/Boolean/src/G4DisplacedSolid.cc:G4Polyhedron* G4DisplacedSolid::GetPolyhedron () const
    ./geometry/solids/Boolean/src/G4IntersectionSolid.cc:  G4Polyhedron* pA = fPtrSolidA->GetPolyhedron();
    ./geometry/solids/Boolean/src/G4IntersectionSolid.cc:  G4Polyhedron* pB = fPtrSolidB->GetPolyhedron();
    ./geometry/solids/Boolean/src/G4SubtractionSolid.cc:  G4Polyhedron* pA = fPtrSolidA->GetPolyhedron();
    ./geometry/solids/Boolean/src/G4SubtractionSolid.cc:  G4Polyhedron* pB = fPtrSolidB->GetPolyhedron();
    ./geometry/solids/Boolean/src/G4UnionSolid.cc:  G4Polyhedron* pA = fPtrSolidA->GetPolyhedron();
    ./geometry/solids/Boolean/src/G4UnionSolid.cc:  G4Polyhedron* pB = fPtrSolidB->GetPolyhedron();
    ./geometry/solids/Boolean/src/G4BooleanSolid.cc:G4Polyhedron* G4BooleanSolid::GetPolyhedron () const
    ./geometry/navigation/src/G4DrawVoxels.cc:       pVVisManager->Draw((*pplist)[i].GetPolyhedron(),
    [blyth@cms01 source]$ 


::

       1048 //* GetPolyhedron -----------------------------------------------------
       1049 
       1050 G4Polyhedron* G4TwistedTubs::GetPolyhedron () const
       1051 {
       1052   if ((!fpPolyhedron) ||
       1053       (fpPolyhedron->GetNumberOfRotationStepsAtTimeOfCreation() !=
       1054        fpPolyhedron->GetNumberOfRotationSteps()))
       1055   {
       1056     delete fpPolyhedron;
       1057     fpPolyhedron = CreatePolyhedron();
       1058   }
       1059   return fpPolyhedron;
       1060 }


::

       1003 G4Polyhedron* G4TwistedTubs::CreatePolyhedron () const
       1004 {
       1005   // number of meshes
       1006   //
       1007   G4double dA = std::max(fDPhi,fPhiTwist);
       1008   const G4int m =
       1009     G4int(G4Polyhedron::GetNumberOfRotationSteps() * dA / twopi) + 2;
       1010   const G4int n =
       1011     G4int(G4Polyhedron::GetNumberOfRotationSteps() * fPhiTwist / twopi) + 2;
       1012 
       1013   const G4int nnodes = 4*(m-1)*(n-2) + 2*m*m ;
       1014   const G4int nfaces = 4*(m-1)*(n-1) + 2*(m-1)*(m-1) ;
       1015 
       1016   G4Polyhedron *ph=new G4Polyhedron;
       1017   typedef G4double G4double3[3];
       1018   typedef G4int G4int4[4];
       1019   G4double3* xyz = new G4double3[nnodes];  // number of nodes 
       1020   G4int4*  faces = new G4int4[nfaces] ;    // number of faces
       1021   fLowerEndcap->GetFacets(m,m,xyz,faces,0) ;
       1022   fUpperEndcap->GetFacets(m,m,xyz,faces,1) ;
       1023   fInnerHype->GetFacets(m,n,xyz,faces,2) ;
       1024   fFormerTwisted->GetFacets(m,n,xyz,faces,3) ;
       1025   fOuterHype->GetFacets(m,n,xyz,faces,4) ;
       1026   fLatterTwisted->GetFacets(m,n,xyz,faces,5) ;
       1027 
       1028   ph->createPolyhedron(nnodes,nfaces,xyz,faces);
       1029 
       1030   delete[] xyz;
       1031   delete[] faces;
       1032 
       1033   return ph;
       1034 }


::

     74 G4Polyhedron* G4CSGSolid::GetPolyhedron () const
     75 {
     76   if (!fpPolyhedron ||
     77       fpPolyhedron->GetNumberOfRotationStepsAtTimeOfCreation() !=
     78       fpPolyhedron->GetNumberOfRotationSteps())
     79     {
     80       delete fpPolyhedron;
     81       fpPolyhedron = CreatePolyhedron();
     82     }
     83   return fpPolyhedron;




CreateSurfaces
----------------

::

   1065 void G4TwistedTubs::CreateSurfaces()
   1066 {
   1067    // create 6 surfaces of TwistedTub
   1068      
   1069    G4ThreeVector x0(0, 0, fEndZ[0]);
   1070    G4ThreeVector n (0, 0, -1);
   1071 
   1072    fLowerEndcap = new G4TwistTubsFlatSide("LowerEndcap",
   1073                                     fEndInnerRadius, fEndOuterRadius,
   1074                                     fDPhi, fEndPhi, fEndZ, -1) ;
   1075 
   1076    fUpperEndcap = new G4TwistTubsFlatSide("UpperEndcap",
   1077                                     fEndInnerRadius, fEndOuterRadius,
   1078                                     fDPhi, fEndPhi, fEndZ, 1) ;
   1079   
   1080    G4RotationMatrix    rotHalfDPhi;
   1081    rotHalfDPhi.rotateZ(0.5*fDPhi);
   ....
   1106    // set neighbour surfaces
   1107    //
   1108    fLowerEndcap->SetNeighbours(fInnerHype, fLatterTwisted,
   1109                                fOuterHype, fFormerTwisted);
   1110    fUpperEndcap->SetNeighbours(fInnerHype, fLatterTwisted,
   1111                                fOuterHype, fFormerTwisted);
   ....
   1120 }



CreatePolyhedron
----------------

::

    [blyth@cms01 source]$ find . -name '*.hh' -exec grep -H CreatePolyhedron {} \;
    ./geometry/management/include/G4VSolid.hh:    virtual G4Polyhedron* CreatePolyhedron () const;
    ./geometry/management/include/G4ReflectedSolid.hh:    G4Polyhedron* CreatePolyhedron () const ;
    ./geometry/solids/BREPS/include/G4BREPSolidPCone.hh:  G4Polyhedron* CreatePolyhedron () const;
    ./geometry/solids/BREPS/include/G4BREPSolid.hh:  G4Polyhedron* CreatePolyhedron () const;
    ./geometry/solids/BREPS/include/G4BREPSolidPolyhedra.hh:  G4Polyhedron* CreatePolyhedron () const;
    ./geometry/solids/CSG/include/G4Orb.hh:    G4Polyhedron* CreatePolyhedron() const;
    ./geometry/solids/CSG/include/G4Trap.hh:    G4Polyhedron* CreatePolyhedron   () const;
    ./geometry/solids/CSG/include/G4Cons.hh:    G4Polyhedron* CreatePolyhedron() const;
    ./geometry/solids/CSG/include/G4Sphere.hh:    G4Polyhedron* CreatePolyhedron() const;
    ./geometry/solids/CSG/include/G4Torus.hh:    G4Polyhedron*       CreatePolyhedron   () const;
    ./geometry/solids/CSG/include/G4Trd.hh:    G4Polyhedron* CreatePolyhedron   () const;
    ./geometry/solids/CSG/include/G4Tubs.hh:// 22.07.96 J.Allison: Changed SendPolyhedronTo to CreatePolyhedron
    ./geometry/solids/CSG/include/G4Tubs.hh:    G4Polyhedron*       CreatePolyhedron   () const;
    ./geometry/solids/CSG/include/G4Para.hh:    G4Polyhedron* CreatePolyhedron   () const;
    ./geometry/solids/CSG/include/G4Box.hh://                     and SendPolyhedronTo() to CreatePolyhedron()
    ./geometry/solids/CSG/include/G4Box.hh:    G4Polyhedron* CreatePolyhedron   () const;
    ./geometry/solids/specific/include/G4TessellatedSolid.hh:    virtual G4Polyhedron* CreatePolyhedron () const;
    ./geometry/solids/specific/include/G4Polycone.hh:  G4Polyhedron* CreatePolyhedron() const;
    ./geometry/solids/specific/include/G4VCSGfaceted.hh:    virtual G4Polyhedron* CreatePolyhedron() const = 0;
    ./geometry/solids/specific/include/G4TwistedTubs.hh:  G4Polyhedron   *CreatePolyhedron   () const;
    ./geometry/solids/specific/include/G4Paraboloid.hh:    G4Polyhedron* CreatePolyhedron() const;
    ./geometry/solids/specific/include/G4EllipticalTube.hh:    G4Polyhedron* CreatePolyhedron() const;
    ./geometry/solids/specific/include/G4Ellipsoid.hh:    G4Polyhedron* CreatePolyhedron() const;
    ./geometry/solids/specific/include/G4Polyhedra.hh:  G4Polyhedron* CreatePolyhedron() const;
    ./geometry/solids/specific/include/G4Hype.hh:  G4Polyhedron* CreatePolyhedron   () const;
    ./geometry/solids/specific/include/G4Tet.hh:    G4Polyhedron* CreatePolyhedron   () const;
    ./geometry/solids/specific/include/G4VTwistedFaceted.hh:  virtual G4Polyhedron   *CreatePolyhedron   () const ;
    ./geometry/solids/specific/include/G4EllipticalCone.hh:    G4Polyhedron* CreatePolyhedron() const;
    ./geometry/solids/Boolean/include/G4DisplacedSolid.hh:    G4Polyhedron* CreatePolyhedron () const ;
    ./geometry/solids/Boolean/include/G4UnionSolid.hh:    G4Polyhedron* CreatePolyhedron () const ;
    ./geometry/solids/Boolean/include/G4SubtractionSolid.hh:    G4Polyhedron* CreatePolyhedron () const ;
    ./geometry/solids/Boolean/include/G4IntersectionSolid.hh:    G4Polyhedron* CreatePolyhedron () const ;



$DYB/external/build/LCG/geant4.9.2.p01/source/geometry/solids/CSG/src/G4Tubs.cc::

   1890 G4Polyhedron* G4Tubs::CreatePolyhedron () const
   1891 {
   1892   return new G4PolyhedronTubs (fRMin, fRMax, fDz, fSPhi, fDPhi) ;
   1893 }

