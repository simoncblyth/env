{

//  gROOT->Reset(); // reset the previous pmt pattern
	
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
  gm->SetPMTHit( 1 , 3. );
  gm->SetPMTHit( 2 , 5. );
  gm->SetPMTHit( 3,  1. );

  gm->ResetPMT();

  gm->SetPMTHit( 5 , 3. );
  gm->SetPMTHit( 6 , 5. );
  gm->SetPMTHit( 7 , 1. );
  gm->SetPMTHit(15 , 4. );

  gm->ResetPMT();

  gm->SetPMTHit( 8 , 2.);
  gm->SetPMTHit( 1 , 5.);
  gm->SetPMTHit( 9 , 0.7);
  
}



