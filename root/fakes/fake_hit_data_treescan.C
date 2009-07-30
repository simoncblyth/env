


TString FormScanString( const TString& vars , const TString& prefix ){
    
    TObjArray* vs = vars.Tokenize(",");
    TString scan = "" ;
    for(Int_t iv=0 ; iv<vs->GetSize() ; ++iv ){
        TObjString* tob = (TObjString*)vs->At(iv);
        if( tob != NULL ){
          scan += Form("%s.%s", prefix.Data(), tob->GetString().Data() );
          if( iv < vs->GetSize() - 1 ){
            scan += ":" ; 
          }
        }
    }
    return scan ;
}

void fake_hit_data_treescan(){
    
    TFile* f = new TFile("dummy.root");
    TTree* event_tree = (TTree*)f->Get("event_tree");

    TString vars = "code,id,wl,weight,x_pmt,y_pmt,z_pmt,px,py,pz,polx,poly,polz,t,producerID";
    TString prefix = "fFakeHitData" ;
    TString scan = FormScanString( vars, prefix );
    event_tree->Scan( scan );


    // code : changed to : the sensitive detector index 
    // id   : ditto .. but reuse for smth else ?
    // 

}




