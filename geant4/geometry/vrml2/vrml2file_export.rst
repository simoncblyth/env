
VRML2FILE Geometry Export
===========================

.. contents:: :local:

TODO
-----

#. prevent culling, 

   * DONE, only World volume was being culled

#. chase export warnings, 

   * DONE, but what to do about them not clear

#. validation that complete geometry exported 

   * seems OK, from dumping of the pv names BUT comparison against a GDML export would be good


GiGa python export
--------------------

Export geometry by adding to `opw/fmcpmuon.py`, using inspiration from  :dybsvn:`source:dybgaudi/trunk/Simulation/DetSim/python/visdet.py`::

        #
        # tone down verbosity of G4RunManager, 
        # skipping the particle table dump for Verbosity less than 3
        # https://wiki.bnl.gov/dayabay/index.php?title=FAQ:How_to_turn_off_long_particle_listing%3F
        #
        from GiGa.GiGaConf import GiGa, GiGaRunManager
        giga = GiGa("GiGa")
        gigarm = GiGaRunManager("GiGa.GiGaMgr")
        gigarm.Verbosity = 2 

        #
        # geometry export attempt
        # 
        from GaussTools.GaussToolsConf import GiGaRunActionCommand
        grac = GiGaRunActionCommand("GiGa.GiGaRunActionCommand")
        grac.BeginOfRunCommands = [ 
             "/vis/open VRML2FILE",
             "/vis/drawVolume",
             "/vis/viewer/flush"
        ]    
        #from GiGa.GiGaConf import GiGa
        giga = GiGa()
        giga.RunAction = grac    
        giga.VisManager = "GiGaVisManager/GiGaVis"



G4VRML2FileViewer::SendViewParameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    116 void G4VRML2FileViewer::SendViewParameters ()
    117 {
    118   // Calculates view representation based on extent of object being
    119   // viewed and (initial) direction of camera.  (Note: it can change
    120   // later due to user interaction via visualization system's GUI.)
    121 
    122 #if defined DEBUG_FR_VIEW
    123       G4cerr << "***** G4VRML2FileViewer::SendViewParameters()\n";
    124 #endif
    125 
    126     // error recovery
    127     if ( fsin_VHA < 1.0e-6 ) { return ; }
    128 
    129     // camera distance
    130     G4double extent_radius = fSceneHandler.GetScene()->GetExtent().GetExtentRadius();
    131     G4double camera_distance = extent_radius / fsin_VHA ;
    132 
    133     // camera position on Z axis
    134     const G4Point3D&    target_point
    135       = fSceneHandler.GetScene()->GetStandardTargetPoint()
    136       + fVP.GetCurrentTargetPoint();
    137     G4double        E_z = target_point.z() + camera_distance;
    138     G4Point3D       E(0.0, 0.0, E_z );
    139 
    140     // VRML codes are generated below   
    141     fDest << G4endl;
    142     fDest << "#---------- CAMERA" << G4endl;
    143     fDest << "Viewpoint {"         << G4endl;
    144     fDest << "\t" << "position "           ;
    145     fDest                 << E.x() << " "  ;
    146     fDest                 << E.y() << " "  ;
    147     fDest                 << E.z() << G4endl ;
    148     fDest << "}" << G4endl;
    149     fDest << G4endl;
    150 
    151 }



G4VRML2FileViewer::DrawView
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

     70 void G4VRML2FileViewer::DrawView()
     71 {
     72 #if defined DEBUG_FR_VIEW
     73     G4cerr << "***** G4VRML2FileViewer::DrawView()" << G4endl;
     74 #endif
     75 
     76     fSceneHandler.VRMLBeginModeling() ;
     77 
     78         // Viewpoint node
     79         SendViewParameters();
     80 
     81     // Here is a minimal DrawView() function.
     82     NeedKernelVisit();
     83     ProcessView();
     84     FinishView();
     85 }



export settings
----------------

Need to avoid culling for a complete export.
Observe that the only volume being culled was the World volume.

::

    /vis/viewer/set/culling global false
    /vis/viewer/set/culling coveredDaughters false 


visualization/management/src/G4VisManager.cc
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    650 void G4VisManager::CreateViewer (G4String name,G4String XGeometry) {
    651 
    652   if (!fInitialised) Initialise ();
    653 
    654   if (fpSceneHandler) {
    655     G4VViewer* p = fpGraphicsSystem -> CreateViewer (*fpSceneHandler, name);
    656 
    ...
    660     G4ViewParameters vp = p->GetViewParameters();
    661     vp.SetXGeometryString(XGeometry);
    662     p->SetViewParameters(vp); //parse string and store parameters
    ...
    680       } else {
    681     fpViewer = p;                             // Make current.
    682     fpSceneHandler -> AddViewerToList (fpViewer);
    683     fpSceneHandler -> SetCurrentViewer (fpViewer);
    684 
    685     if (fVerbosity >= confirmations) {
    686       G4cout << "G4VisManager::CreateViewer: new viewer created."
    687          << G4endl;
    688     }
    689 
    690     const G4ViewParameters& vp = fpViewer->GetViewParameters();
    691     if (fVerbosity >= parameters) {
    692       G4cout << " view parameters are:\n  " << vp << G4endl;
    693     }
    694
    695     if (vp.IsCulling () && vp.IsCullingInvisible ()) {
    696       static G4bool warned = false;
    697       if (fVerbosity >= confirmations) {
    698         if (!warned) {
    699           G4cout <<
    700   "NOTE: objects with visibility flag set to \"false\""
    701   " will not be drawn!"
    702   "\n  \"/vis/viewer/set/culling global false\" to Draw such objects."
    703   "\n  Also see other \"/vis/viewer/set\" commands."
    704              << G4endl;
    705           warned = true;
    706         }
    707       }
    708     }
    709     if (vp.IsCullingCovered ()) {
    710       static G4bool warned = false;
    711       if (fVerbosity >= warnings) {
    712         if (!warned) {
    713           G4cout <<
    714   "WARNING: covered objects in solid mode will not be rendered!"
    715   "\n  \"/vis/viewer/set/culling coveredDaughters false\" to reverse this."
    716   "\n  Also see other \"/vis/viewer/set\" commands."
    717              << G4endl;
    718           warned = true;
    719         }
    720       }

::

    /vis/sceneHandler/create VRML2FILE
    G4VisManager::SetCurrentGraphicsSystem: system now VRML2FILE
    Graphics system set to VRML2FILE
    New scene handler "scene-handler-0" created.
    /vis/viewer/create ! ! 600 

    G4VisManager::CreateViewer: new viewer created.
     view parameters are:
      View parameters and options:
      Drawing style: wireframe
      Auxiliary edges: invisible
      Representation style: polyhedron
      Culling: on
      Culling invisible objects: on
      Density culling: off 
      Culling daughters covered by opaque mothers: off 
      Section flag: false
      No cutaway planes
      Explode factor: 1 about centre: (0,0,0)
      No. of sides used in circle polygon approximation: 24
      Viewpoint direction:  (0,0,1)
      Up vector:            (0,1,0)
      Field half angle:     0   
      Zoom factor:          1
      Scale factor:         (1,1,1)
      Current target point: (0,0,0)
      Dolly distance:       0
      Light does not move with camera
      Relative lightpoint direction: (1,1,1)
      Actual lightpoint direction: (1,1,1)
      Derived parameters for standard view of object of unit radius:
        Camera distance:   1
        Near distance:     1e-06
        Far distance:      2
        Front half height: 1
      Default VisAttributes:
      G4VisAttributes: visible, daughters visible, colour: (1,1,1,1)
      linestyle: solid, line width: 1
      drawing style: not forced, auxiliary edge visibility: not forced
      line segments per circle: not forced.
      time range: (-1.79769e+308,1.79769e+308)
      G4AttValue pointer is zero, G4AttDef pointer is zero
      Default TextVisAttributes:
      G4VisAttributes: visible, daughters visible, colour: (0,0,1,1)
      linestyle: solid, line width: 1
      drawing style: not forced, auxiliary edge visibility: not forced
      line segments per circle: not forced.
      time range: (-1.79769e+308,1.79769e+308)
      G4AttValue pointer is zero, G4AttDef pointer is zero
      Default marker: G4VMarker: position: (0,0,0), world size: 0, screen size: 5
               fill style: no fill
               No Visualization Attributes
      Global marker scale: 1
      Global lineWidth scale: 1
      Marker not hidden by surfaces.
      Window size hint: 600x600
      X geometry string: 600x600
      Auto refresh: false
      Background colour: (0,0,0,1)
      Picking requested: false

    NOTE: objects with visibility flag set to "false" will not be drawn!
      "/vis/viewer/set/culling global false" to Draw such objects.
      Also see other "/vis/viewer/set" commands.

    /////////   end from G4VisManager::CreateViewer 

    New viewer "viewer-0 (VRML2FILE)" created.
    Issue /vis/viewer/refresh to see effect.
    /vis/scene/create
    New empty scene "scene-0" created.
    /vis/scene/add/volume world
    First occurrence of "Universe"
      found at depth 0,
      with a requested depth of further descent of <0 (unlimited),
      has been added to scene "scene-0".
    /vis/sceneHandler/attach
    Scene "scene-0" attached to scene handler "scene-handler-0.
      (You may have to refresh with "/vis/viewer/flush" if view is not "auto-refresh".)
    NOTE: For systems which are not "auto-refresh" you will need to
      issue "/vis/viewer/refresh" or "/vis/viewer/flush".
    /vis/viewer/refresh viewer-0
    Refreshing viewer "viewer-0 (VRML2FILE)"...
    ===========================================
    Output VRML 2.0 file: g4_00.wrl
    Maximum number of files in the destination directory: 100
      (Customizable with the environment variable: G4VRMLFILE_MAX_FILE_NUM)
    ===========================================


where is the export steered from
------------------------------------

::

    [blyth@cms01 source]$ find visualization -name '*.cc' -exec grep -H vis/drawVolume {} \;
    visualization/management/src/G4VisCommandsCompound.cc:  UImanager->ApplyCommand(G4String("/vis/drawVolume " + pvname));
    visualization/management/src/G4VisCommandsCompound.cc:////////////// /vis/drawVolume ///////////////////////////////////////
    visualization/management/src/G4VisCommandsCompound.cc:  fpCommand = new G4UIcmdWithAString("/vis/drawVolume", this);
    visualization/management/src/G4VisCommandsSceneAdd.cc:       << "\n  /vis/drawVolume " << name
    visualization/management/src/G4VisManager.cc:     "\n  Null scene pointer. Use \"/vis/drawVolume\" or"
    visualization/management/src/G4VisCommandsViewer.cc:    "\n  (or use compound command \"/vis/drawVolume\")."
    visualization/management/src/G4VisCommandsViewer.cc:    "\n  (or use compound command \"/vis/drawVolume\")."
    visualization/RayTracer/src/G4RayTracerSceneHandler.cc:  // "/vis/drawVolume"...

::

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


export warnings
----------------

Total of ~220 warning lines.::

    120 Traversing scene data...
    121 BooleanProcessor::triangulateContour : could not generate a triangle (infinite loop)
    122 BooleanProcessor: boolean operation failed
    123 BooleanProcessor::triangulateFace : too small contour
    124 BooleanProcessor::triangulateContour : could not generate a triangle (infinite loop)
    125 BooleanProcessor: boolean operation failed
    126 BooleanProcessor::triangulateContour : could not generate a triangle (infinite loop)
    127 BooleanProcessor: boolean operation failed
    ...
    151 BooleanProcessor::triangulateContour : could not generate a triangle (infinite loop)
    152 BooleanProcessor: boolean operation failed
    153 BooleanProcessor::assembleFace(74) : could not find next edge of the contour
    154 BooleanProcessor: boolean operation failed
    155 BooleanProcessor::caseIE : unimplemented case
    156 BooleanProcessor::caseIE : unimplemented case
    ...
    218 BooleanProcessor::caseIE : unimplemented case
    219 BooleanProcessor: boolean operation failed
    220 BooleanProcessor::execute : unknown faces !!!
    221 BooleanProcessor: boolean operation failed
    222 BooleanProcessor::execute : unknown faces !!!
    223 BooleanProcessor: boolean operation failed
    224 BooleanProcessor::triangulateFace : too small contour
    225 BooleanProcessor::triangulateContour : could not generate a triangle (infinite loop)
    226 BooleanProcessor: boolean operation failed
    227 BooleanProcessor::execute : unknown faces !!!
    228 BooleanProcessor: boolean operation failed
    229 BooleanProcessor::caseIE : unimplemented case
    ...
    335 BooleanProcessor::caseIE : unimplemented case
    336 BooleanProcessor::caseIE : unimplemented case
    337 BooleanProcessor: boolean operation failed
    338 Viewer "viewer-0 (VRML2FILE)" refreshed.
    339   (You might also need "/vis/viewer/update".)
    340 /vis/viewer/update viewer-0
    341 Viewer "viewer-0 (VRML2FILE)" post-processing triggered.
    342 *** VRML 2.0 File  g4_00.wrl  is generated.


But from examining the `graphics_reps/src/BooleanProcessor.src`, the *boolean operation failed* 
marks PROCESSOR_ERROR returns from `HepPolyhedron BooleanProcessor::execute`.  
There are only 45 of those::

    simon:geant4 blyth$ grep boolean\ operation\ failed export.txt | wc -l
          45


::

    786 void BooleanProcessor::caseIE(ExtEdge &, ExtEdge &)
    787 /***********************************************************************
    788  *                                                                     *
    789  * Name: BooleanProcessor::caseIE                    Date:    19.01.00 *
    790  * Author: E.Chernyaev                               Revised:          *
    791  *                                                                     *
    792  * Function: Intersection/Edge-touch case                              *
    793  *                                                                     *
    794  ***********************************************************************/
    795 {
    796   processor_error = 1;
    797   std::cout
    798     << "BooleanProcessor::caseIE : unimplemented case"
    799     << std::endl;
    800 }


::

    (gdb) b 'BooleanProcessor::caseIE(ExtEdge&, ExtEdge&)'
    Breakpoint 1 at 0xb55591a8: file src/BooleanProcessor.src, line 796.
    (gdb) c
    Continuing.
    ...
    ===========================================
    Output VRML 2.0 file: g4_04.wrl
    Maximum number of files in the destination directory: 100
      (Customizable with the environment variable: G4VRMLFILE_MAX_FILE_NUM) 
    ===========================================
    Traversing scene data...
    BooleanProcessor::triangulateContour : could not generate a triangle (infinite loop)
    BooleanProcessor: boolean operation failed
    BooleanProcessor::triangulateFace : too small contour
    BooleanProcessor::triangulateContour : could not generate a triangle (infinite loop)
    BooleanProcessor: boolean operation failed
    BooleanProcessor::triangulateContour : could not generate a triangle (infinite loop)
    BooleanProcessor: boolean operation failed
    BooleanProcessor::triangulateContour : could not generate a triangle (infinite loop)
    BooleanProcessor: boolean operation failed
    BooleanProcessor::triangulateContour : could not generate a triangle (infinite loop)
    BooleanProcessor: boolean operation failed
    BooleanProcessor::triangulateContour : could not generate a triangle (infinite loop)
    BooleanProcessor: boolean operation failed
    BooleanProcessor::triangulateContour : could not generate a triangle (infinite loop)
    BooleanProcessor: boolean operation failed
    BooleanProcessor::triangulateContour : could not generate a triangle (infinite loop)
    BooleanProcessor: boolean operation failed
    BooleanProcessor::triangulateFace : too small contour
    BooleanProcessor::triangulateContour : could not generate a triangle (infinite loop)
    BooleanProcessor: boolean operation failed
    BooleanProcessor::triangulateContour : could not generate a triangle (infinite loop)
    BooleanProcessor: boolean operation failed
    BooleanProcessor::triangulateContour : could not generate a triangle (infinite loop)
    BooleanProcessor: boolean operation failed
    BooleanProcessor::triangulateContour : could not generate a triangle (infinite loop)
    BooleanProcessor: boolean operation failed
    BooleanProcessor::triangulateContour : could not generate a triangle (infinite loop)
    BooleanProcessor: boolean operation failed
    BooleanProcessor::triangulateContour : could not generate a triangle (infinite loop)
    BooleanProcessor: boolean operation failed
    BooleanProcessor::triangulateContour : could not generate a triangle (infinite loop)
    BooleanProcessor: boolean operation failed
    BooleanProcessor::assembleFace(74) : could not find next edge of the contour
    BooleanProcessor: boolean operation failed

    Breakpoint 1, BooleanProcessor::caseIE (this=0xb5575620) at src/BooleanProcessor.src:796
    796       processor_error = 1;
    Current language:  auto; currently c++
    (gdb) bt
    #0  BooleanProcessor::caseIE (this=0xb5575620) at src/BooleanProcessor.src:796
    #1  0xb5559487 in BooleanProcessor::testFaceVsFace (this=0xb5575620, iface1=32, iface2=129) at src/BooleanProcessor.src:873
    #2  0xb555cd6f in BooleanProcessor::execute (this=0xb5575620, op=0, a=@0xfba7d18, b=@0xfba7d40) at src/BooleanProcessor.src:1888
    #3  0xb555d172 in HepPolyhedron::add (this=0xfba7d18, p=@0xfba7d40) at src/HepPolyhedron.cc:2269
    #4  0x04ee9019 in G4UnionSolid::CreatePolyhedron (this=0xb612c90) at src/G4UnionSolid.cc:459
    #5  0x04ee0c12 in G4BooleanSolid::GetPolyhedron (this=0xb612c90) at src/G4BooleanSolid.cc:229
    #6  0xb6b0df34 in G4VSceneHandler::RequestPrimitives (this=0xfb58a90, solid=@0xb612c90) at src/G4VSceneHandler.cc:449
    #7  0xb6b0c99b in G4VSceneHandler::AddSolid (this=0xfb58a90, solid=@0xb612c90) at src/G4VSceneHandler.cc:252
    #8  0x0560f889 in G4VRML2FileSceneHandler::AddSolid (this=0xfb58a90, vsolid=@0xb612c90) at src/G4VRML2SceneHandlerFunc.icc:78
    #9  0x04ee7eb1 in G4UnionSolid::DescribeYourselfTo (this=0xb612c90, scene=@0xfb58a90) at src/G4UnionSolid.cc:446
    #10 0xb59df186 in G4PhysicalVolumeModel::DescribeSolid (this=0xfb58fd0, theAT=@0xbfc99d50, pSol=0xb612c90, pVisAttribs=0xfb5acd0, sceneHandler=@0xfb58a90) at src/G4PhysicalVolumeModel.cc:551
    #11 0xb59e0bf2 in G4PhysicalVolumeModel::DescribeAndDescend (this=0xfb58fd0, pVPV=0xb390228, requestedDepth=-12, pLV=0xb39ddb0, pSol=0xb612c90, pMaterial=0xb2e57b8, theAT=@0xbfc9a530, sceneHandler=@0xfb58a90) at src/G4PhysicalVolumeModel.cc:456
    #12 0xb59dfba3 in G4PhysicalVolumeModel::VisitGeometryAndGetVisReps (this=0xfb58fd0, pVPV=0xb390228, requestedDepth=-12, theAT=@0xbfc9a530, sceneHandler=@0xfb58a90) at src/G4PhysicalVolumeModel.cc:222
    #13 0xb59e0dc1 in G4PhysicalVolumeModel::DescribeAndDescend (this=0xfb58fd0, pVPV=0xb5932e0, requestedDepth=-11, pLV=0xb2e7bb0, pSol=0xb5fd350, pMaterial=0xb472c58, theAT=@0xbfc9ad10, sceneHandler=@0xfb58a90) at src/G4PhysicalVolumeModel.cc:523
    #14 0xb59dfba3 in G4PhysicalVolumeModel::VisitGeometryAndGetVisReps (this=0xfb58fd0, pVPV=0xb5932e0, requestedDepth=-11, theAT=@0xbfc9ad10, sceneHandler=@0xfb58a90) at src/G4PhysicalVolumeModel.cc:222
    #15 0xb59e0dc1 in G4PhysicalVolumeModel::DescribeAndDescend (this=0xfb58fd0, pVPV=0xb2c3620, requestedDepth=-10, pLV=0xb57a940, pSol=0xb629f10, pMaterial=0xb2e57b8, theAT=@0xbfc9b4f0, sceneHandler=@0xfb58a90) at src/G4PhysicalVolumeModel.cc:523
    #16 0xb59dfba3 in G4PhysicalVolumeModel::VisitGeometryAndGetVisReps (this=0xfb58fd0, pVPV=0xb2c3620, requestedDepth=-10, theAT=@0xbfc9b4f0, sceneHandler=@0xfb58a90) at src/G4PhysicalVolumeModel.cc:222
    #17 0xb59e0dc1 in G4PhysicalVolumeModel::DescribeAndDescend (this=0xfb58fd0, pVPV=0xb4721e8, requestedDepth=-9, pLV=0xb469910, pSol=0xb468d98, pMaterial=0xb488ff8, theAT=@0xbfc9bcd0, sceneHandler=@0xfb58a90) at src/G4PhysicalVolumeModel.cc:523
    #18 0xb59dfba3 in G4PhysicalVolumeModel::VisitGeometryAndGetVisReps (this=0xfb58fd0, pVPV=0xb4721e8, requestedDepth=-9, theAT=@0xbfc9bcd0, sceneHandler=@0xfb58a90) at src/G4PhysicalVolumeModel.cc:222
    ...
    (gdb) frame 10
    #10 0xb59df186 in G4PhysicalVolumeModel::DescribeSolid (this=0xfb58fd0, theAT=@0xbfc99d50, pSol=0xb612c90, pVisAttribs=0xfb5acd0, sceneHandler=@0xfb58a90) at src/G4PhysicalVolumeModel.cc:551
    551         pSol -> DescribeYourselfTo (sceneHandler);  // Standard treatment.
    (gdb) p this->GetCurrentLV()
    $1 = (G4LogicalVolume *) 0xb39ddb0
    (gdb) p this->GetCurrentLV()->GetName()
    $2 = {<std::basic_string<char,std::char_traits<char>,std::allocator<char> >> = {static npos = 4294967295, _M_dataplus = {<std::allocator<char>> = {<__gnu_cxx::new_allocator<char>> = {<No data fields>}, <No data fields>}, 
          _M_p = 0xb39dccc "/dd/Geometry/PMT/lvHeadonPmtMount"}}, <No data fields>}
    (gdb) 

    (gdb) frame 11
    #11 0xb59e0bf2 in G4PhysicalVolumeModel::DescribeAndDescend (this=0xfb58fd0, pVPV=0xb390228, requestedDepth=-12, pLV=0xb39ddb0, pSol=0xb612c90, pMaterial=0xb2e57b8, theAT=@0xbfc9a530, sceneHandler=@0xfb58a90) at src/G4PhysicalVolumeModel.cc:456
    456         DescribeSolid (theNewAT, pSol, pVisAttribs, sceneHandler);

    (gdb) p pLV
    $3 = (G4LogicalVolume *) 0xb39ddb0

    (gdb) p pLV->GetName()
    $4 = {<std::basic_string<char,std::char_traits<char>,std::allocator<char> >> = {static npos = 4294967295, _M_dataplus = {<std::allocator<char>> = {<__gnu_cxx::new_allocator<char>> = {<No data fields>}, <No data fields>}, 
          _M_p = 0xb39dccc "/dd/Geometry/PMT/lvHeadonPmtMount"}}, <No data fields>}

    (gdb) p pVPV
    $5 = (class G4VPhysicalVolume *) 0xb390228

    (gdb) p pVPV->GetName()
    $6 = (const G4String &) @0xb39024c: {<std::basic_string<char,std::char_traits<char>,std::allocator<char> >> = {static npos = 4294967295, _M_dataplus = {<std::allocator<char>> = {<__gnu_cxx::new_allocator<char>> = {<No data fields>}, <No data fields>}, 
          _M_p = 0xb3901dc "/dd/Geometry/AD/lvOIL#pvAdPmtArray#pvAd2inPmt:1#pvHeadonPmtMount"}}, <No data fields>}


Another caseIE points to same PV LV (maybe duplicate named or other face pairs of the same solid)::

    (gdb) frame 11
    #11 0xb59e0bf2 in G4PhysicalVolumeModel::DescribeAndDescend (this=0xfb58fd0, pVPV=0xb390228, requestedDepth=-12, pLV=0xb39ddb0, pSol=0xb612c90, pMaterial=0xb2e57b8, theAT=@0xbfc9a530, sceneHandler=@0xfb58a90) at src/G4PhysicalVolumeModel.cc:456
    456         DescribeSolid (theNewAT, pSol, pVisAttribs, sceneHandler);

    (gdb) p pVPV->GetName()
    $7 = (const G4String &) @0xb39024c: {<std::basic_string<char,std::char_traits<char>,std::allocator<char> >> = {static npos = 4294967295, _M_dataplus = {<std::allocator<char>> = {<__gnu_cxx::new_allocator<char>> = {<No data fields>}, <No data fields>}, 
          _M_p = 0xb3901dc "/dd/Geometry/AD/lvOIL#pvAdPmtArray#pvAd2inPmt:1#pvHeadonPmtMount"}}, <No data fields>}

    (gdb) p pLV->GetName()
    $8 = {<std::basic_string<char,std::char_traits<char>,std::allocator<char> >> = {static npos = 4294967295, _M_dataplus = {<std::allocator<char>> = {<__gnu_cxx::new_allocator<char>> = {<No data fields>}, <No data fields>}, 
          _M_p = 0xb39dccc "/dd/Geometry/PMT/lvHeadonPmtMount"}}, <No data fields>}

    (gdb) p this->GetCurrentDescription()
    $13 = {<std::basic_string<char,std::char_traits<char>,std::allocator<char> >> = {static npos = 4294967295, _M_dataplus = {<std::allocator<char>> = {<__gnu_cxx::new_allocator<char>> = {<No data fields>}, <No data fields>}, 
          _M_p = 0xfba7d74 "G4PhysicalVolumeModel /dd/Geometry/AD/lvOIL#pvAdPmtArray#pvAd2inPmt:1#pvHeadonPmtMount.1"}}, <No data fields>}


Continuing on for 4 or 5 more caseIE, getting a diffent name::

    (gdb) frame 11
    #11 0xb59e0bf2 in G4PhysicalVolumeModel::DescribeAndDescend (this=0xfb58fd0, pVPV=0xb2a42a8, requestedDepth=-12, pLV=0xb285d28, pSol=0xb285c10, pMaterial=0xb2e57b8, theAT=@0xbfc9a530, sceneHandler=@0xfb58a90) at src/G4PhysicalVolumeModel.cc:456
    456         DescribeSolid (theNewAT, pSol, pVisAttribs, sceneHandler);
    (gdb) p this->GetCurrentDescription()
    $16 = {<std::basic_string<char,std::char_traits<char>,std::allocator<char> >> = {static npos = 4294967295, _M_dataplus = {<std::allocator<char>> = {<__gnu_cxx::new_allocator<char>> = {<No data fields>}, <No data fields>}, 
          _M_p = 0xfd46e94 "G4PhysicalVolumeModel /dd/Geometry/AD/lvOIL#pvSstTopHub.1464"}}, <No data fields>}
    (gdb) 



Huh thats confusing, the pVPV is not changing as move to different frame, need to use the raw pointer::

    (gdb) frame 13
    #13 0xb59e0dc1 in G4PhysicalVolumeModel::DescribeAndDescend (this=0xfb58fd0, pVPV=0xb5932e0, requestedDepth=-11, pLV=0xb2e7bb0, pSol=0xb5fd350, pMaterial=0xb472c58, theAT=@0xbfc9ad10, sceneHandler=@0xfb58a90) at src/G4PhysicalVolumeModel.cc:523
    523             (pVPV, requestedDepth - 1, theNewAT, sceneHandler);
    (gdb) p pVPV->GetName()
    $18 = (const G4String &) @0xb2a42cc: {<std::basic_string<char,std::char_traits<char>,std::allocator<char> >> = {static npos = 4294967295, _M_dataplus = {<std::allocator<char>> = {<__gnu_cxx::new_allocator<char>> = {<No data fields>}, <No data fields>}, 
          _M_p = 0xb275b24 "/dd/Geometry/AD/lvOIL#pvSstTopHub"}}, <No data fields>}
    (gdb) p pVPV
    $19 = (class G4VPhysicalVolume *) 0xb2a42a8
    (gdb) p ((class G4VPhysicalVolume *) 0xb5932e0)->GetName()
    $20 = (const G4String &) @0xb593304: {<std::basic_string<char,std::char_traits<char>,std::allocator<char> >> = {static npos = 4294967295, _M_dataplus = {<std::allocator<char>> = {<__gnu_cxx::new_allocator<char>> = {<No data fields>}, <No data fields>}, 
          _M_p = 0xb5932ac "/dd/Geometry/AD/lvSST#pvOIL"}}, <No data fields>}
    (gdb) 



vis recursion
~~~~~~~~~~~~~~~

Need to follow the recursion to make sense of this.


* $DYB/external/build/LCG/geant4.9.2.p01/source/visualization/modeling/include/G4PhysicalVolumeModel.hh

$DYB/external/build/LCG/geant4.9.2.p01/source/visualization/modeling/src/G4PhysicalVolumeModel.cc::

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




::

    (gdb) b 'BooleanProcessor::execute(int, HepPolyhedron const&, HepPolyhedron const&)'
    Breakpoint 3 at 0xb59f34dc: file src/BooleanProcessor.src, line 1788.

    (gdb) c
    Continuing.

    Breakpoint 3, BooleanProcessor::execute (this=0xb5a0c620, op=0, a=@0x1024af10, b=@0xfb72a98) at src/BooleanProcessor.src:1788
    1788      processor_error = 0;
    (gdb) bt
    #0  BooleanProcessor::execute (this=0xb5a0c620, op=0, a=@0x1024af10, b=@0xfb72a98) at src/BooleanProcessor.src:1788
    #1  0xb59f4172 in HepPolyhedron::add (this=0x1024af10, p=@0xfb72a98) at src/HepPolyhedron.cc:2269
    #2  0x04592019 in G4UnionSolid::CreatePolyhedron (this=0xbb625d0) at src/G4UnionSolid.cc:459
    #3  0x04589c12 in G4BooleanSolid::GetPolyhedron (this=0xbb625d0) at src/G4BooleanSolid.cc:229
    #4  0xb709af34 in G4VSceneHandler::RequestPrimitives (this=0x1024ab10, solid=@0xbb625d0) at src/G4VSceneHandler.cc:449
    #5  0xb709999b in G4VSceneHandler::AddSolid (this=0x1024ab10, solid=@0xbb625d0) at src/G4VSceneHandler.cc:252
    #6  0x046c8889 in G4VRML2FileSceneHandler::AddSolid (this=0x1024ab10, vsolid=@0xbb625d0) at src/G4VRML2SceneHandlerFunc.icc:78
    #7  0x04590eb1 in G4UnionSolid::DescribeYourselfTo (this=0xbb625d0, scene=@0x1024ab10) at src/G4UnionSolid.cc:446
    #8  0xb5ef6186 in G4PhysicalVolumeModel::DescribeSolid (this=0x1024b0c0, theAT=@0xbfabaa50, pSol=0xbb625d0, pVisAttribs=0x1024cdc8, sceneHandler=@0x1024ab10) at src/G4PhysicalVolumeModel.cc:551
    #9  0xb5ef7bf2 in G4PhysicalVolumeModel::DescribeAndDescend (this=0x1024b0c0, pVPV=0xba84708, requestedDepth=-3, pLV=0xbcfa7e0, pSol=0xbb625d0, pMaterial=0xbb62c88, theAT=@0xbfabb230, sceneHandler=@0x1024ab10) at src/G4PhysicalVolumeModel.cc:456
    #10 0xb5ef6ba3 in G4PhysicalVolumeModel::VisitGeometryAndGetVisReps (this=0x1024b0c0, pVPV=0xba84708, requestedDepth=-3, theAT=@0xbfabb230, sceneHandler=@0x1024ab10) at src/G4PhysicalVolumeModel.cc:222
    #11 0xb5ef7dc1 in G4PhysicalVolumeModel::DescribeAndDescend (this=0x1024b0c0, pVPV=0xba26fd0, requestedDepth=-2, pLV=0xbb61578, pSol=0xbb7cce0, pMaterial=0xbb7e670, theAT=@0xbfabba10, sceneHandler=@0x1024ab10) at src/G4PhysicalVolumeModel.cc:523
    #12 0xb5ef6ba3 in G4PhysicalVolumeModel::VisitGeometryAndGetVisReps (this=0x1024b0c0, pVPV=0xba26fd0, requestedDepth=-2, theAT=@0xbfabba10, sceneHandler=@0x1024ab10) at src/G4PhysicalVolumeModel.cc:222
    #13 0xb5ef7dc1 in G4PhysicalVolumeModel::DescribeAndDescend (this=0x1024b0c0, pVPV=0xbc0f528, requestedDepth=-1, pLV=0xba26c80, pSol=0xba26c00, pMaterial=0xbe623f8, theAT=@0xbfabbf80, sceneHandler=@0x1024ab10) at src/G4PhysicalVolumeModel.cc:523
    #14 0xb5ef6ba3 in G4PhysicalVolumeModel::VisitGeometryAndGetVisReps (this=0x1024b0c0, pVPV=0xbc0f528, requestedDepth=-1, theAT=@0xbfabbf80, sceneHandler=@0x1024ab10) at src/G4PhysicalVolumeModel.cc:222
    #15 0xb5ef8219 in G4PhysicalVolumeModel::DescribeYourselfTo (this=0x1024b0c0, sceneHandler=@0x1024ab10) at src/G4PhysicalVolumeModel.cc:170
    #16 0xb709a9f2 in G4VSceneHandler::ProcessScene (this=0x1024ab10) at src/G4VSceneHandler.cc:516
    #17 0xb70a162b in G4VViewer::ProcessView (this=0x1024cc58) at src/G4VViewer.cc:122
    #18 0x046c980c in G4VRML2FileViewer::DrawView (this=0x1024cc58) at src/G4VRML2FileViewer.cc:83


export output
-----------------

.. literalinclude:: export.txt

booleanprocessor
~~~~~~~~~~~~~~~~~~

::

    [blyth@cms01 source]$ find . -name '*.*' -exec grep -H BooleanProcessor {} \;
    ./visualization/management/include/G4VSceneHandler.hh:  // Generic clipping using the BooleanProcessor in graphics_reps is
    ./visualization/OpenGL/src/G4OpenGLSceneHandler.cc:  // when the BooleanProcessor is up to it, abandon this and use
    ./visualization/OpenGL/src/G4OpenGLSceneHandler.cc:  // But...if not, when the BooleanProcessor is up to it...
    ./visualization/OpenGL/src/G4OpenGLViewer.cc:  // BooleanProcessor is up to it, abandon this and use generic
    ./graphics_reps/src/HepPolyhedron.cc:#include "BooleanProcessor.src"
    ./graphics_reps/src/HepPolyhedron.cc:static BooleanProcessor processor;
    ./graphics_reps/src/BooleanProcessor.src: * Name: BooleanProcessor                            Date:    10.12.99 *
    ./graphics_reps/src/BooleanProcessor.src://                       its members from the BooleanProcessor class ---
    ./graphics_reps/src/BooleanProcessor.src:  friend class BooleanProcessor;
    ./graphics_reps/src/BooleanProcessor.src:class BooleanProcessor {
    ./graphics_reps/src/BooleanProcessor.src:  BooleanProcessor() {}
    ./graphics_reps/src/BooleanProcessor.src:  ~BooleanProcessor() {}
    ./graphics_reps/src/BooleanProcessor.src:void BooleanProcessor::takePolyhedron(const HepPolyhedron & p,
    ./graphics_reps/src/BooleanProcessor.src: * Name: BooleanProcessor::takePolyhedron            Date:    16.12.99 *
    ./graphics_reps/src/BooleanProcessor.src:double BooleanProcessor::findMinMax()
    ./graphics_reps/src/BooleanProcessor.src: * Name: BooleanProcessor::findMinMax                Date:    16.12.99 *
    ./graphics_reps/src/BooleanProcessor.src:void BooleanProcessor::selectOutsideFaces(int & ifaces, int & iout)
    ./graphics_reps/src/BooleanProcessor.src: * Name: BooleanProcessor::selectOutsideFaces        Date:    10.01.00 *
    ./graphics_reps/src/BooleanProcessor.src:int BooleanProcessor::testFaceVsPlane(ExtEdge & edge)
     ...

scene scaling
~~~~~~~~~~~~~~

Blender has trouble with the large extent, maybe scaling can be done while still in G4

visualization/management/src/G4VSceneHandler.cc

::

    271 void G4VSceneHandler::AddPrimitive (const G4Scale& scale) {
    272 
    273   const G4double margin(0.01);
    274   // Fractional margin - ensures scale is comfortably inside viewing
    275   // volume.
    276   const G4double oneMinusMargin (1. - margin);
    277 
    278   const G4VisExtent& sceneExtent = fpScene->GetExtent();
    279 
    280   // Useful constants...
    281   const G4double length(scale.GetLength());
    282   const G4double halfLength(length / 2.);
    283   const G4double tickLength(length / 20.);
    284   const G4double piBy2(halfpi);
    285 
    286   // Get size of scene...
    287   const G4double xmin = sceneExtent.GetXmin();
    288   const G4double xmax = sceneExtent.GetXmax();
    289   const G4double ymin = sceneExtent.GetYmin();
    290   const G4double ymax = sceneExtent.GetYmax();
    291   const G4double zmin = sceneExtent.GetZmin();
    292   const G4double zmax = sceneExtent.GetZmax();



examine the g4_00.wrl comments 
--------------------------------

Comments provide the names of solids. 

::

    00001 #VRML V2.0 utf8
       02 # Generated by VRML 2.0 driver of GEANT4
       03 #---------- CAMERA
       04 #---------- SOLID: /dd/Structure/Sites/db-rock.1000
       05 #---------- SOLID: /dd/Geometry/Sites/lvNearSiteRock#pvNearHallTop.1000
       06 #---------- SOLID: /dd/Geometry/Sites/lvNearHallTop#pvNearTopCover.1000
       07 #---------- SOLID: /dd/Geometry/Sites/lvNearHallTop#pvNearTeleRpc#pvNearTeleRpc:1.1
       08 #---------- SOLID: /dd/Geometry/RPC/lvRPCMod#pvRPCFoam.1000
       09 #---------- SOLID: /dd/Geometry/RPC/lvRPCFoam#pvBarCham14Array#pvBarCham14ArrayOne:1#pvBarCham14Unit.1
       10 #---------- SOLID: /dd/Geometry/RPC/lvRPCBarCham14#pvRPCGasgap14.1000
       11 #---------- SOLID: /dd/Geometry/RPC/lvRPCGasgap14#pvStrip14Array#pvStrip14ArrayOne:1#pvStrip14Unit.1
       12 #---------- SOLID: /dd/Geometry/RPC/lvRPCGasgap14#pvStrip14Array#pvStrip14ArrayOne:2#pvStrip14Unit.2
       13 #---------- SOLID: /dd/Geometry/RPC/lvRPCGasgap14#pvStrip14Array#pvStrip14ArrayOne:3#pvStrip14Unit.3
    ...
    12227 #---------- SOLID: /dd/Geometry/Sites/lvNearHallBot#pvNearHallRadSlabs#pvNearHallRadSlab4.1004
    12228 #---------- SOLID: /dd/Geometry/Sites/lvNearHallBot#pvNearHallRadSlabs#pvNearHallRadSlab5.1005
    12229 #---------- SOLID: /dd/Geometry/Sites/lvNearHallBot#pvNearHallRadSlabs#pvNearHallRadSlab6.1006
    12230 #---------- SOLID: /dd/Geometry/Sites/lvNearHallBot#pvNearHallRadSlabs#pvNearHallRadSlab7.1007
    12231 #---------- SOLID: /dd/Geometry/Sites/lvNearHallBot#pvNearHallRadSlabs#pvNearHallRadSlab8.1008
    12232 #---------- SOLID: /dd/Geometry/Sites/lvNearHallBot#pvNearHallRadSlabs#pvNearHallRadSlab9.1009
    12233 #End of file.



The names are not distinct::

    simon:samples blyth$ grep ^# g4_00.wrl > comments.txt

    simon:samples blyth$ wc -l comments.txt 
    12233 comments.txt
    simon:samples blyth$ cat comments.txt | sort | uniq | wc -l 
        5646

All comments are SOLID apart from 4 lines, one of which is CAMERA::

    simon:samples blyth$ grep SOLID g4_00.wrl | wc -l 
       12229

All geometry use `IndexedFaceSet`::

    simon:samples blyth$ grep geometry g4_00.wrl | uniq
                    geometry IndexedFaceSet {

All material use `Material`::

    simon:samples blyth$ grep material g4_00.wrl | sort | uniq
                            material Material {


Writing polyhedron primitives
-------------------------------

::

    169 void G4VRML2SCENEHANDLER::AddPrimitive(const G4Polyhedron& polyhedron)
    170 {
    171 #if defined DEBUG_SCENE_FUNC
    172     G4cerr << "***** AddPrimitive(G4Polyhedron)" << "\n";
    173 #endif
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
    191     if ( IsPVPickable() ) {
    192 
    193      fDest << "Anchor {" << "\n";
    194      fDest << " description " << "\"" << pv_name << "\"" << "\n";
    195      fDest << " url \"\" " << "\n";
    196      fDest << " children [" << "\n";
    197     }
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
    211         point.transform( *fpObjectTransformation );
    212 
    213         fDest << "\t\t\t\t\t";
    214         fDest <<                   point.x() << " ";
    215         fDest <<                   point.y() << " ";
    216         fDest <<                   point.z() << "," << "\n";
    217     }
    218     fDest << "\t\t\t\t" <<      "]" << "\n"; // point
    219     fDest << "\t\t\t"   << "}"      << "\n"; // coord
    220 
    221     fDest << "\t\t\t"   << "coordIndex [" << "\n";
    222 
    223     // facet loop
    224     G4int f;
    225     for (f = polyhedron.GetNoFacets(); f; f--) {
    226 
    227         // edge loop
    228         G4bool notLastEdge;
    229         G4int index = -1, edgeFlag = 1;
    230         fDest << "\t\t\t\t";
    231         do {
    232             notLastEdge = polyhedron.GetNextVertexIndex(index, edgeFlag);
    233             fDest << index - 1 << ", ";
    234         } while (notLastEdge);
    235         fDest << "-1," << "\n";
    236     }
    237     fDest << "\t\t\t"   << "]" << "\n"; // coordIndex
    238 
    239     fDest << "\t\t\t"   << "solid FALSE" << "\n"; // draw backfaces
    240 
    241     fDest << "\t\t" << "}"     << "\n"; // IndexFaceSet
    242     fDest << "\t" << "}"       << "\n"; // Shape
    243 
    244     if ( IsPVPickable() ) {
    245      fDest << " ]"              << "\n"; // children
    246      fDest << "}"               << "\n"; // Anchor
    247     }
    248 
    249 }


vrml2file driver 
-----------------


::


    [blyth@cms01 src]$ l G4VRML2*
    -rw-r--r--  1 blyth blyth  2954 Mar 16  2009 G4VRML2.cc
    -rw-r--r--  1 blyth blyth  2588 Mar 16  2009 G4VRML2File.cc
    -rw-r--r--  1 blyth blyth  7184 Mar 16  2009 G4VRML2FileSceneHandler.cc
    -rw-r--r--  1 blyth blyth  4932 Mar 16  2009 G4VRML2FileViewer.cc
    -rw-r--r--  1 blyth blyth  4407 Mar 16  2009 G4VRML2SceneHandler.cc
    -rw-r--r--  1 blyth blyth 18480 Mar 16  2009 G4VRML2SceneHandlerFunc.icc
    -rw-r--r--  1 blyth blyth  4651 Mar 16  2009 G4VRML2Viewer.cc
    [blyth@cms01 src]$ 
    [blyth@cms01 src]$ pwd
    /data/env/local/dyb/trunk/external/build/LCG/geant4.9.2.p01/source/visualization/VRML/src

    [blyth@cms01 src]$ grep icc G4VRML2*.cc
    G4VRML2FileSceneHandler.cc:#include "G4VRML2SceneHandlerFunc.icc"
    G4VRML2SceneHandler.cc:#include "G4VRML2SceneHandlerFunc.icc"


Common 



source/visualization/VRML/include/G4VRML2FileSceneHandler.hh
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

From Geant4 solid instances into the scene rep.::

     46 class G4VRML2FileSceneHandler: public G4VSceneHandler {
     47 
     48   friend class G4VRML2FileViewer;
     49 
     50 // methods (public) 
     51 public:
     52     G4VRML2FileSceneHandler(G4VRML2File& system, const G4String& name = "");
     53     virtual ~G4VRML2FileSceneHandler();
     54     void AddSolid(const G4Box&);
     55     void AddSolid(const G4Cons&);
     56     void AddSolid(const G4Tubs&);

Pickability, what does that mean for VRML2 ?::

    103     // PV name pickability  
    104     if( getenv( "G4VRML_PV_PICKABLE" ) != NULL ) {
    105 
    106         int is_pickable ;
    107         sscanf( getenv("G4VRML_PV_PICKABLE"), "%d", &is_pickable ) ;
    108 
    109         if ( is_pickable ) { SetPVPickability ( true ) ; }
    110     }
    111 
    112     // PV Transparency
    113     SetPVTransparency ();

Defer to common code for decomposition into primitives::

    111 void G4VRML2SCENEHANDLER::AddSolid(const G4Sphere& sphere)
    112 {
    113 #if defined DEBUG_SCENE_FUNC
    114     G4cerr << "***** AddSolid sphere" << "\n" ;
    115 #endif
    116     VRMLBeginModeling () ;
    117     G4VSceneHandler::AddSolid(sphere) ;
    118 }

Preprocessor icc trick to avoid duplication::

    127 #define  G4VRML2SCENEHANDLER   G4VRML2FileSceneHandler
    128 #define  IS_CONNECTED   this->isConnected() 
    129 #include "G4VRML2SceneHandlerFunc.icc"
    130 #undef   IS_CONNECTED
    131 #undef   G4VRML2SCENEHANDLER


The header.

::

    498 void G4VRML2SCENEHANDLER::VRMLBeginModeling()
    499 {
    500     if (!IS_CONNECTED ) {
    501 #if defined DEBUG_SCENE_FUNC
    502     G4cerr << "***** VRMLBeginModeling (started)" << "\n" ;
    503 #endif
    504         this->connectPort();
    505         fDest << "#VRML V2.0 utf8" << "\n";
    506         fDest << "# Generated by VRML 2.0 driver of GEANT4\n" << "\n";
    507     }
    508 }



