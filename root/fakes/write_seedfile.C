
//
//   prepare a list of seeds in an xml file   
//   invoke with eg
//       .x ../root/write_seedfile.C("10k.root", "xml/no_fake_hits.xml" , 0 )
//

int write_seedfile(TString rootfile, TString xmlpath , Int_t nfake ){
    
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
    
    TString description="events with no fake hits" ;
    TString path = Form("%s/%s",gSystem->Getenv("DAYA_DATA_DIR"),xmlpath.Data());
    cout << "writing seed file to " << path << endl ;
    
    ofstream o(path.Data());
    o << Form("<seedfile rootfile=\"%s\" xmlpath=\"%s\" description=\"%s\" >\n",  rootfile.Data(), xmlpath.Data(), description.Data() ) ;
    o << Form("<!-- /dyw/run/randomSeedFile %s  ## to read this seedfile -->\n", xmlpath.Data() );
    
    Int_t n(0);
    for (Int_t i=0;i<imax;i++) {
        t->GetEntry(i);                 
        if( i%100 == 0 ) cout << i << " " ; 
        // replace with the selection you are interested in 
        if( evt->GetNofFakeHits(-1) == nfake ){
            o << Form("<seed n=\"%d\" >%d</seed>\n",n++, evt->ranSeed)  ;
        }
    }
    cout << endl ;
    o <<  Form("<!-- /run/beamOn %d ## to run over all seeds -->\n" , n );
    o << "</seedfile>\n" ;
    o.close();
    return 0 ;
}