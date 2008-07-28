
//
//   prepare a list of seeds in an xml file   
//   invoke with eg
//       .x ../root/write_seedfile.C("10k.root", "xml/no_fake_hits.xml" , 0 )
//
//   this file is to prepare seeds about different nFake


//void select(void*);

void thho_write_seedfile_transmission_nFake(TString rootfile, TString xmlpath ){
    
    gSystem->Load("$DYW/InstallArea/$CMTCONFIG/lib/libMCEvent.so");
    dywGLEvent* evt = new dywGLEvent();

    TFile *f = new TFile( rootfile ,"read");
    if( f->IsZombie() ) {
        cout << "ABORT cannot open " << rootfile << endl; 
        return 1 ;
    }
    TTree* t = (TTree*) f->Get("event_tree");
    t->SetBranchAddress("dayabay_MC_event_output",&evt);
    
    Int_t nevent = t->GetEntries();
    Int_t imax = nevent ;
    //imax = 1000 ;
    
    TString description="events with no fake hits" ;
    TString path = Form("%s/%s",gSystem->Getenv("DAYA_DATA_DIR"),xmlpath.Data());
    cout << "writing seed file to " << path << endl ;
    
    ofstream o(path.Data());
    o << Form("<seedfile rootfile=\"%s\" xmlpath=\"%s\" description=\"%s\" >\n",  rootfile.Data(), xmlpath.Data(), description.Data() ) ;
    o << Form("<!-- /dyw/run/randomSeedFile %s  ## to read this seedfile -->\n", xmlpath.Data() );
    
    Int_t n(0);
    for (Int_t i=0;i<imax;i++) {
        t->GetEntry(i);
        if( i%100 == 0 ) cout << i << endl ; 

	// replace with the selection you are interested in
	
        //get vertex data
        Float_t vertx(0), verty(0), vertz(0);
  	TClonesArray &vert = *(evt->GetVertexArray());
  	((dywGLVertex*)vert[0])->GetPosition(vertx,verty,vertz);
  
  	//get position, time, weight ,and id data
	Float_t xf(0), yf(0), zf(0);
  	Float_t tf(0) , wf(0) ;
  	Int_t idf(0);

	TClonesArray &fha = *(evt->GetFakeHitArray());
  	const Int_t nFake = fha.GetEntries(); 
	dywGLPhotonHitData* phd = NULL ;
  
	if(nFake == 2){

	  o << Form("<seed n=\"%d\" >%d</seed>\n",n++, evt->ranSeed)  ;
}

}
    cout << endl ;
    o <<  Form("<!-- /run/beamOn %d ## to run over all seeds -->\n" , n );
    o << "</seedfile>\n" ;
    o.close();

}
