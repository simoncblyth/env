{

    using namespace TMath;

 // 
  // Reads Aberdeen_World.root plus a G4dyb output root file with PMTPositionMap
  //      writes World.root and WorldWithPMTs.root
  //   
  //
  // imports a gGeoManager ending up with a closed geometry , as written by VGM with PMTs excluded
  // as PMTs cannot yet be exported from Geant4 

   TGeoManager::Import("Aberdeen_World.root")  ;

  // Export the volume allows more flexibility to be able to add to the 
  // geometry tree  before closing it.
   gGeoManager->GetTopVolume()->Export("World.root") ; // export the world volume for flexibility
  

  // load the PMT positions ... 
   gROOT->ProcessLine(".L PMTMap.C");
   PMTMap* pm = new PMTMap ;
   pm->Load("abergeo.root");

   // import the world that was just exported ... this time can add to the geometry
   TGeoVolume* top = TGeoVolume::Import("World.root", "World");
   gGeoManager->SetTopVolume( top );
   gGeoManager->CloseGeometry();

   TGeoNode* tn = top->GetNode(0) ;  // Worldneutron_log_0 


   TEveManager::Create();
   TEveGeoTopNode* etn = new TEveGeoTopNode(gGeoManager, tn );
   gEve->AddGlobalElement(etn);


   gStyle->SetPalette(1, 0);

   TEveRGBAPalette* pal = new TEveRGBAPalette(0, 130);
   
   TEveBoxSet* cones = new TEveBoxSet("ConeSet");
   cones->SetPalette(pal);
   cones->Reset(TEveBoxSet::kBT_Cone, kFALSE, 64);
  
   TEveVector dir, pos;

  // add some cones to represent the PMTs 

   Double_t rad = 10 ;  // radius
   Double_t val = rad ;
   Double_t height = 10 ;
   Double_t sc = 0.1 ; 

   PMT* pj = NULL ;
   for( Int_t j = 0 ; j < pm->fEntries ; j++ ){
       pj = pm->fPMT[j] ;

       Double_t  phi = TMath::RadToDeg()*TMath::ATan2(  pj->y , pj->x );  
       Double_t theta = 0.; 
       cout << Form(" %d %f %f %f %f (deg) " , pj->id , pj->x , pj->y , pj->z , phi ) << endl ;

       pos.Set( pj->x , pj->y , pj->z );
       pos*= sc ;
       dir.Set(Cos(phi)*Cos(theta), Sin(phi)*Cos(theta), Sin(theta));
       dir*= height ;

       cones->AddCone(pos, dir, rad);
       cones->DigitValue(val );
  }

   cones->RefitPlex();
   TEveTrans& t = cones->RefMainTrans(); 
   t.SetPos(0, 0, 0);
  
   gEve->AddElement(cones);
   gEve->Redraw3D(kTRUE);

   

  // Retain flexibility by exporting the modified world volume rather than the manager..
  // ... this way subsequent code can just access the volume 
  //  without concern for this nasty  PMT fix 
  //  
  // top->Export("WorldWithPMTs.root");



  // See the below from the TGeo expert :
  //      http://root.cern.ch/root/roottalk/roottalk03/5255.html
  //   To represent PMT response, you could add some more boxes whose size/color/visibility
  //   you change for each event. 
  //   NB : create the heirarchy of volumes in the beginning and subsequently 
  //       just change sizes/attributes 
  //         ... more efficient plus will avoid memory management problems
  //
  //
  // Continue manipulations with :
  //
  //  TGeoVolume* top = TGeoVolume::Import("WorldWithPMTs.root", "World"); 
  //  gGeoManager->SetTopVolume( top )
  //  gGeoManager->CloseGeometry()
  //  top->Draw("ogle")
  //


}


