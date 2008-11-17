{
    using namespace TMath;
  // Reads Aberdeen_World.root plus a G4dyb output root file with PMTPositionMap
  //      writes World.root and WorldWithPMTs.root
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
   gNav = gGeoManager->GetCurrentNavigator();


   cout << "loading GeoMap.. " << endl ;
   gROOT->ProcessLine(".L GeoMap.C");
   gm = new GeoMap(tn) ;

//   gm->SetVisibility("steeltank_log_0", kFALSE );

   gm->SetLineColor("1m_Plastic_Scintillator_log_[0-9]*", kRed );
   gm->SetLineColor("2m_Proportional_Tube_Gas_[0-9]*", kGreen );
   gm->SetLineColor("1.5m_Plastic_Scintillator_log_[0-9]*", kBlue );
//   gm->SetLineColor("2m_Proportional_Tube_Gas_[0-9]*", kCyan );
   gm->SetLineColor("2m_Plastic_Scintillator_log_[0-9]*", kMagenta );

   gm->SetVisibility("logicMovalbeTF_0", kFALSE );
   gm->SetVisibility("logicMovalbeTF_0x", kFALSE );

   gm->SetVisibility("Door_0",kFALSE);
   gm->SetVisibility("LeadPlateTop_0", kFALSE);
   gm->SetVisibility("LeadPlateBottom_0", kFALSE);
  
   gm->SetTransparency("outer_log_0", 50 ) ;


   TEveManager::Create();
   TEveGeoTopNode* etn = new TEveGeoTopNode(gGeoManager, tn );
   gEve->AddGlobalElement(etn);

   gStyle->SetPalette(1, 0);

   TEveCompound* cmp = new TEveCompound;
   cmp->SetMainColor(kGreen);
   gEve->AddElement(cmp);
   cmp->OpenCompound();

   TEveVector dir, pos;

  // add some cones to represent the PMTs 

   Double_t rad = 10 ;  // radius
   Double_t val = rad ;
   Double_t height = 10 ;
   Double_t sc = 0.1 ; 
   
   TEveRGBAPalette* pal = new TEveRGBAPalette(0, 130);

   PMT* pj = NULL ;
   for( Int_t j = 0 ; j < pm->fEntries ; j++ ){
       pj = pm->fPMT[j] ;

       // this is inside in order to be able to select the PMTs indiviually 
       TEveBoxSet* cones = new TEveBoxSet(Form("PMT%0.2d",j));
       cones->SetPalette(pal);
       cones->Reset(TEveBoxSet::kBT_Cone, kFALSE, 64);
 
       Double_t  phi = TMath::RadToDeg()*TMath::ATan2(  pj->y , pj->x );  
       Double_t theta = 0.; 
       cout << Form(" %d %f %f %f %f (deg) " , pj->id , pj->x , pj->y , pj->z , phi ) << endl ;

       pos.Set( pj->x , pj->y , pj->z );
       pos*= sc ;
       dir.Set(Cos(phi)*Cos(theta), Sin(phi)*Cos(theta), Sin(theta));
       dir*= height ;

       cones->AddCone(pos, dir, rad);
       cones->DigitValue(val );
      //  TEveDigitSet::RefitPlex  Instruct underlying memory allocator to regroup itself into a contiguous memory chunk.
       cones->RefitPlex();
       
     //  gEve->AddElement(cones);
       cmp->AddElement(cones); 
 }

 cmp->CloseCompound();

 //  TEveElement::RefMainTrans   Return reference to main transformation. It is created if not yet existing.
 //  TEveTrans& t = cones->RefMainTrans(); 
 //  t.SetPos(0, 0, 0);
  
 gEve->Redraw3D(kTRUE);


}


