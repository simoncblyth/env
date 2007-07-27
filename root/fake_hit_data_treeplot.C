{
    TFile* f = new TFile("dummy.root");
    TTree* event_tree = (TTree*)f->Get("event_tree");
    
    TString name = "fake_hit_data" ;
    TCanvas* c = new TCanvas( name , name );
    
    event_tree->Draw("fAke.x_pmt:fAke.y_pmt");
 
     // the id specififies which FakeSD skin ...
    event_tree->Draw("fAke.x_pmt:fAke.y_pmt:fAke.z_pmt","fAke.id==0");
    event_tree->Draw("fAke.x_pmt:fAke.y_pmt:fAke.z_pmt","fAke.id==1");
   
    event_tree->Draw("fAke.t","fAke.id==0");
    event_tree->Draw("fAke.t","fAke.id==1");    
   
    event_tree->Draw("fAke.wl:fAke.t");
   
    event_tree->Draw("GetNofFakeHits(-1)"); // fake hits in all FakeSD
    event_tree->Draw("GetNofFakeHits(0)");  // fake hits in FakeSD 0
    
    event_tree->Draw("fAke.code" ) ;       // zero based code, of hit numbers within an event in all FakeSD 
    event_tree->Draw("fAke.code:fAke.t");
    
    
    
    
    
    
    
    // number of fake hits in all FakeSD ...
    event_tree->Draw("GetNofFakeHits(-1)"); 
    
    // the multiple reflections, cause 
    //
    // if the fAke.code == fAke.id then its a hit from an outgoing track ... with no reflection issues
    // in which case the difference between the entries ... of 0 or 1 gives transmission and propagation between em 
    //  approx 8 percent loss
    //
    event_tree->Draw("fAke.code","fAke.code==fAke.id"); 
    
    
    event_tree->Draw("fAke.code", "fAke.code==0") ; // subtubs...   1955
    event_tree->Draw("fAke.code", "fAke.code==1")  ; // subtubs ... 1799
    
    
    event_tree->Scan("ranSeed", "GetNofFakeHits(-1)==0");
        
    /*
        ************************
        *    Row   *   ranSeed *
        ************************
        *        0 *         0 *
        *        3 *         3 *
        *        7 *         7 *
        *       14 *        14 *
        *       17 *        17 *
        *       20 *        20 *
    */
    
    
    // need the totals ... can the event_tree->Draw plot the results of a GetFakeSDHits( isd ) ????
    
    
    //event_tree->Draw("fAke.wl:fAke.t");

     // want to know the 


}