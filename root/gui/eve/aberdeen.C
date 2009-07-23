{

gROOT->ProcessLine(" .L $ENV_HOME/aberdeen/root/GeoMap.C");

TString src = "$ENV_HOME/aberdeen/root/Aberdeen_World.root" ;
//TString src = "$ENV_HOME/aberdeen/root/WorldWithPMTs.root" ;

GeoMap* gm = new GeoMap() ;  // not working other .root yet 
gm->ImportVolume( src ,"World");
    
TGeoNode* tn = gm->GetNod("World_1");
TEveGeoTopNode* etn = new TEveGeoTopNode(gGeoManager, tn );

TEveManager::Create();  // bring up the Eve GUI
gEve->AddGlobalElement(etn);
gEve->Redraw3D(kTRUE);

gm->SetVisibility("^.*$", kFALSE);       
gm->SetVisibility("Tube", kTRUE );

   

  // gGeoManager = gEve->GetGeometry("$ENV_HOME/aberdeen/root/Aberdeen_World.root");

   
  // TGeoVolume* tv = gGeoManager->GetTopVolume();
  // TGeoNode* tn = gGeoManager->GetTopNode();
  // TEveGeoTopNode* etn = new TEveGeoTopNode(gGeoManager, tn);
  // gEve->AddGlobalElement(etn);

   // do we need to add all nodes to Eve one by one to have picking control of em ???

  //  gEve->FullRedraw3D(kTRUE);

   // EClipType not exported to CINT (see TGLUtil.h):
   // 0 - no clip, 1 - clip plane, 2 - clip box
   //  gEve->GetGLViewer()->GetClipSet()->SetClipType(1);
   // gEve->GetGLViewer()->RefreshPadEditor(gEve->GetGLViewer());
}


