DAE cf WRL at code level
==========================


Compare G4DAE and VRML2 geometry handling code
------------------------------------------------

#. comparing VRML2 and G4DAE code for vertices : looks identical,

   * maybe some parameters : dont think so, all seem at defaults
   * precision issue 
   
.. sidebar:: Promising explanation but seemingly not the case 

   DAE creation so far uses expedient of running from a Geant4 geometry created from an exported GDML file, for development speed. 
   **BUT** that compounds precision issues.  The polyhedron creation algorithm appears sensitive to precise geometry especially
   when you have subtraction/union solids.
   Checked this by testing DAE creation direct from original in memory model, not the one loaded from the GDML. This 
   allows to compare apples-to-apples rather than comparison against 2nd generation geometry filtered thru GDML precision.
   
   The results of that comparison are precisely the same, perhaps some parameter tweaks in VRML2 ?


BooleanProcessor
----------------

``graphics_reps/src/BooleanProcessor.src`` 



G4Polyhedron::SetNumberOfRotationSteps ?
--------------------------------------------

Given that the differences are all in subtraction/union solids it seems unlikely to be 
a difference in such a parameter.  To determine perhaps could add some ``extra`` metadata
to the exported DAE with param values ? 


::

    [blyth@belle7 source]$ find . -exec grep -H G4Polyhedron:: {} \;
    ./visualization/modeling/src/G4PhysicalVolumeModel.cc:      G4Polyhedron::SetNumberOfRotationSteps
    ./visualization/modeling/src/G4PhysicalVolumeModel.cc:      G4Polyhedron::SetNumberOfRotationSteps(fpMP->GetNoOfSides());
    ./visualization/modeling/src/G4PhysicalVolumeModel.cc:    G4Polyhedron::ResetNumberOfRotationSteps();
    ./visualization/management/src/G4VSceneHandler.cc:    G4Polyhedron::SetNumberOfRotationSteps (GetNoOfSides (fpVisAttribs));
    ./visualization/management/src/G4VSceneHandler.cc:    G4Polyhedron::ResetNumberOfRotationSteps ();
    ./geometry/solids/specific/src/G4TwistedTubs.cc:    G4int(G4Polyhedron::GetNumberOfRotationSteps() * dA / twopi) + 2;
    ./geometry/solids/specific/src/G4TwistedTubs.cc:    G4int(G4Polyhedron::GetNumberOfRotationSteps() * fPhiTwist / twopi) + 2;
    ./geometry/solids/specific/src/G4VTwistedFaceted.cc:    G4int(G4Polyhedron::GetNumberOfRotationSteps() * fPhiTwist / twopi) + 2;
    ./geometry/solids/specific/src/G4Polycone.cc:          G4int(G4Polyhedron::GetNumberOfRotationSteps()
    ./geometry/solids/specific/History:  G4Polyhedron::GetNumberOfRotationSteps().
    ./graphics_reps/include/HepPolyhedron.h://    G4Polyhedron::SetNumberOfRotationSteps
    ./graphics_reps/include/HepPolyhedron.h://    G4Polyhedron::ResetNumberOfRotationSteps ();
    ./graphics_reps/src/G4Polyhedron.cc:G4Polyhedron::G4Polyhedron ():
    ./graphics_reps/src/G4Polyhedron.cc:G4Polyhedron::~G4Polyhedron () {}
    ./graphics_reps/src/G4Polyhedron.cc:G4Polyhedron::G4Polyhedron (const HepPolyhedron& from)
    ./graphics_reps/History:- Added G4Polyhedron::Transform and G4Polyhedron::InvertFacets (Evgeni
    [blyth@belle7 source]$ 


``graphics_reps/include/HepPolyhedron.h``::

    105 //   GetNumberOfRotationSteps()   - get number of steps for whole circle;
    106 //   SetNumberOfRotationSteps (n) - set number of steps for whole circle;
    107 //   ResetNumberOfRotationSteps() - reset number of steps for whole circle
    108 //                            to default value;
    109 //   IsErrorBooleanProcess()- true if there has been an error during the
    110 //                            processing of a Boolean operation.
    ...
    168 #ifndef HEP_POLYHEDRON_HH
    169 #define HEP_POLYHEDRON_HH
    170 
    171 #include <CLHEP/Geometry/Point3D.h>
    172 #include <CLHEP/Geometry/Normal3D.h>
    173 
    174 #ifndef DEFAULT_NUMBER_OF_STEPS
    175 #define DEFAULT_NUMBER_OF_STEPS 24
    176 #endif


``LCG/geant4.9.2.p01/source/visualization/management/src/G4VSceneHandler.cc``::

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



::

    859 G4int G4VSceneHandler::GetNoOfSides(const G4VisAttributes* pVisAttribs)
    860 {
    861   // No. of sides (lines segments per circle) is normally determined
    862   // by the view parameters, but it can be overriddden by the
    863   // ForceLineSegmentsPerCircle in the vis attributes.
    864   G4int lineSegmentsPerCircle = fpViewer->GetViewParameters().GetNoOfSides();
    865   if (pVisAttribs) {
    866     if (pVisAttribs->IsForceLineSegmentsPerCircle())
    867       lineSegmentsPerCircle = pVisAttribs->GetForcedLineSegmentsPerCircle();
    868     const G4int nSegmentsMin = 12;
    869     if (lineSegmentsPerCircle < nSegmentsMin) {
    870       lineSegmentsPerCircle = nSegmentsMin;
    871       G4cout <<
    872     "G4VSceneHandler::GetNoOfSides: attempt to set the"
    873     "\nnumber of line segements per circle < " << nSegmentsMin
    874          << "; forced to " << lineSegmentsPerCircle << G4endl;
    875     }
    876   }
    877   return lineSegmentsPerCircle;
    878 }




Geant4
-------


geometry/solids/Boolean/src/G4UnionSolid.cc::

    453 G4Polyhedron*
    454 G4UnionSolid::CreatePolyhedron () const
    455 {
    456   G4Polyhedron* pA = fPtrSolidA->GetPolyhedron();
    457   G4Polyhedron* pB = fPtrSolidB->GetPolyhedron();
    458   if (pA && pB) {
    459     G4Polyhedron* resultant = new G4Polyhedron (pA->add(*pB));
    460     return resultant;
    461   } else {
    462     std::ostringstream oss;
    463     oss << GetName() <<
    464       ": one of the Boolean components has no corresponding polyhedron.";
    465     G4Exception("G4UnionSolid::CreatePolyhedron",
    466         "", JustWarning, oss.str().c_str());
    467     return 0;
    468   }
    469 }

geometry/solids/Boolean/src/G4SubtractionSolid.cc::

    466 G4Polyhedron*
    467 G4SubtractionSolid::CreatePolyhedron () const
    468 {
    469   G4Polyhedron* pA = fPtrSolidA->GetPolyhedron();
    470   G4Polyhedron* pB = fPtrSolidB->GetPolyhedron();
    471   if (pA && pB)
    472   {
    473     G4Polyhedron* resultant = new G4Polyhedron (pA->subtract(*pB));
    474     return resultant;
    475   }
    476   else
    477   {
    478     std::ostringstream oss;
    479     oss << "Solid - " << GetName()
    480         << " - one of the Boolean components has no" << G4endl
    481         << " corresponding polyhedron. Returning NULL !";
    482     G4Exception("G4SubtractionSolid::CreatePolyhedron()", "InvalidSetup",
    483                 JustWarning, oss.str().c_str());
    484     return 0;
    485   }
    486 }


