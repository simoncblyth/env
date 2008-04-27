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

  //SetPMTHit(Int_t pmt_no,Double_t hitpattern size)
  // 0 <= hitpattern size >= 5.
  gm->SetPMTHit( 0 , 3 );
  
  //  gm->refresh("World_1");
  //refresh the display
  gm->GetVol("World_1")->Draw("ogle");
  
}



