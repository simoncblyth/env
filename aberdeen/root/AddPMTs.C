{

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


   TGeoMaterial* mat = new TGeoMaterial("Vacuum",0,0,0);
   TGeoMedium*   med = new TGeoMedium("Vacuum",1,mat); 

   Double_t sc = 0.1 ;
   Double_t sx = 5. ;
   Double_t sy = 1. ;
   Double_t sz = 5. ;

  // add some boxes to represent the PMTs 

   PMT* pj = NULL ;
   for( Int_t j = 0 ; j < pm->fEntries ; j++ ){
       pj = pm->fPMT[j] ;

       Double_t  phi = TMath::RadToDeg()*TMath::ATan2(  pj->y , pj->x );  
       cout << Form(" %d %f %f %f %f (deg) " , pj->id , pj->x , pj->y , pj->z , phi ) << endl ;

       // represent the PMT with box , flat in y 
       TGeoVolume* pmt = gGeoManager->MakeBox("PMT",med, sx , sy , sz );  
       pmt->SetLineColor(kMagenta);
       pmt->SetVisibility(kTRUE);   

       TGeoRotation* rot =new TGeoRotation();
       rot->SetAngles( 0.,0., phi + 90 ); // first rotate about Z with angle phi, then about the rotated ...

       top->AddNode( pmt , pj->id , new TGeoCombiTrans( sc*pj->x,sc*pj->y,sc*pj->z ,rot  ) ); // first rotate then translate 
   }

   gGeoManager->SetTopVolume( top );
   gGeoManager->CloseGeometry();


   top->Draw("ogle");
   

  // Retain flexibility by exporting the modified world volume rather than the manager..
  // ... this way subsequent code can just access the volume 
  //  without concern for this nasty  PMT fix 
  //  
   top->Export("WorldWithPMTs.root");



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


