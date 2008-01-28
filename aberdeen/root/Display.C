{
  gROOT->ProcessLine(".L GeoMap.C");
  gm = new GeoMap();
  gm->ImportVolume("WorldWithPMTs.root","World");
  gm->GetVol("World_1")->Draw("ogle");
  gm->SelectKeys("^.*PMT.*$");
  gm->SelectKeys("^.*PMT.*$")->ls();


  gm->SetVisibility("^.*$", kFALSE );  // everything invisible
  gm->SetVisibility("PMT", kTRUE );
  gm->SetVisibility("Tube", kTRUE );

  gm->SetPMTHit( 0 , 0.5 );
  


}



