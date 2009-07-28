/*
   This is major refactoring of the example : 
       $ROOTSYS/tutorials/eve/alice_esd_split.C
   see the introductory text there

   The motivation behind the refactoring is to split off experiment 
   specific parts in order to cast the example into a fit state
   to act as a starting point for a quick and dirty event display

*/

R__EXTERN TEveProjectionManager *gRPhiMgr;
R__EXTERN TEveProjectionManager *gRhoZMgr;
TEveGeoShape *gGeoShape;
TGTextEntry *gTextEntry;
TGHProgressBar *gProgress;

// it seems these need to be global ??? 
// otherwise segmentation on setting branch addresses for the trees

TFile* esd_file          = 0;
TFile* esd_friends_file  = 0;
TTree* esd_tree          = 0 ; 

//______________________________________________________________________________
void alice_esd_split()
{

   // just class definitions
   gROOT->ProcessLine(".x EvReader.C");
   gROOT->ProcessLine(".x AliEvReader.C");
   gROOT->ProcessLine(".x EvNav.C");
   gROOT->ProcessLine(".x EvDisp.C");

   IEvReader* evr = EvReader::GetEvReader();
   evr->InitProject();

   EvDisp* evd = EvDisp::GetEvDisp();
   evd->make_gui();
   
   evr->LoadGeometry();  // must come after initializing Eve (?)

   evd->import_projection_geometry();
   evd->load_event();
   evd->update_projections();

   gEve->Redraw3D(kTRUE); // Reset camera after the first event has been shown.
}


