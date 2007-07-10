{
    TFile* f = new TFile("dummy.root");
    TTree* event_tree = (TTree*)f->Get("event_tree");
    
    TString name = "fake_hit_data" ;
    TCanvas* c = new TCanvas( name , name );
    
    event_tree->Draw("fFakeHitData.x_pmt:fFakeHitData.y_pmt");
 
    event_tree->Draw("fFakeHitData.x_pmt:fFakeHitData.y_pmt:fFakeHitData.z_pmt","fFakeHitData.id==0");
    event_tree->Draw("fFakeHitData.x_pmt:fFakeHitData.y_pmt:fFakeHitData.z_pmt","fFakeHitData.id==1");
        
    event_tree->Draw("fFakeHitData.t","fFakeHitData.id==0");
    event_tree->Draw("fFakeHitData.t","fFakeHitData.id==1");    
 
    event_tree->Draw("fFakeHitData.wl:fFakeHitData.t");
    
    // need the totals ... can the event_tree->Draw plot the results of a GetFakeSDHits( isd ) ????
    
    
    //event_tree->Draw("fAke.wl:fFakeHitData.t");
}