
void fake_hit_data_clonesarray(){
    
        gSystem->Load("$DYW/InstallArea/$CMTCONFIG/libMCEvent.so");
        TFile *f = new TFile( "10k.root" ,"read");
        
        dywGLEvent* evt = new dywGLEvent();
        TTree *tree = (TTree*) f->Get("event_tree");
        tree->SetBranchAddress("dayabay_MC_event_output",&evt);
        
        Int_t nevent = tree->GetEntries();
        
        
        for (Int_t i=0;i<100;i++) {
            tree->GetEntry(i);                  //read complete event in memory
            Float_t vertx(0), verty(0), vertz(0);
            
            cout << evt->Summary(2) << endl ;
            

            TClonesArray &vert = *(evt->GetVertexArray());
            ((dywGLVertex*)vert[0])->GetPosition(vertx,verty,vertz);
             //cout<<Form("**** Event %d vertex (%.1f, %.1f, %.1f)",i+1, vertx, verty, vertz)<<endl;
            
            TClonesArray &fha = *(evt->GetFakeHitArray());
            const Int_t nFake = fha.GetEntries();
           
            
            
            
            Float_t xf(0), yf(0), zf(0);
            Float_t tf(0) , wf(0) ;
            Int_t idf(0);
            
            dywGLPhotonHitData* phd = NULL ;
            
            for(size_t ii=0; ii<nFake; ii++){
                phd = (dywGLPhotonHitData*)fha[ii];
                
                phd->GetPosition(xf,yf,zf);
                idf = phd->GetPMTID();
                tf  = phd->GetHitTime();
                wf  = phd->GetWeight();
                
               // cout<<Form(" [%d](%.1f,%.1f,%.1f ; %.1f , %.3f )", idf , xf,yf,zf, tf , wf ) <<endl;
            }
            
            
            TClonesArray &pmt = *(evt->GetPMTHitArray(1));//Get Module_1
            
            const Int_t npmt = pmt.GetEntries();
            Float_t xcathode(0), ycathode(0), zcathode(0);
        
            for(size_t jj=0; jj<npmt; jj++){
                phd = (dywGLPhotonHitData*)pmt[jj];
                
                phd->GetPosition(xcathode,ycathode,zcathode);        
              
                TString ss= Form("PMT%d fires, time %.1f ns, cathode pos (%.1f, %.1f, %.1f)", phd->GetPMTID(), phd->GetHitTime(), xcathode, ycathode, zcathode);
                //cout<<ss<<endl;
            }
        }
}


/*
 
 TFile* f = new TFile("dummy.root");
 TTree* event_tree = (TTree*)f->Get("event_tree");
 TBranch* b = event_tree->GetBranch("fFakeHitData");
 
 TClonesArray* c = NULL ;
 b->SetAddress(&c);
 
 TClonesArray& ca = *c ;
 
 Int_t nevent = event_tree->GetEntries();
 
 for(Int_t i=0 ; i < nevent ; ++i ){
     event_tree->GetEvent(i);   
     
 }
 
 

 root [53] c->GetClass()
 (const class TClass*)0x56501f0
 root [54] c->GetName() 
 (const char* 0x56aff00)"dywGLPhotonHitDatas"
 root [55] 
 root [55] c->GetCurrentCollection()
 (class TCollection*)0x5a3ed80
 root [56] c->GetCurrentCollection()->ls()
 
 
 
 root [30] event_tree->Draw("fFakeHitData.GetEntries()") 
Error in <TTreeFormula::DefinedVariable>: Unknown method:GetEntries() in dywGLPhotonHitData
Error in <TTreeFormula::Compile>:  Bad numerical expression : "fFakeHitData.GetEntries()"
 
 
 // equalizes the counts 
  event_tree->Draw("fFakeHitData.GetPMTID()","fFakeHitData.weight<-0.61") 
 
 // too few bounce back ...as indicated by 
   fFakeHitData.weight > 0 
 
 
 */