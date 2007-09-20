
// counts No. of optical photon at diffenent positions to 
// get the gross "transmission"
// 
// scantrees():scan all trees, select, and write into a new file
// clculatetrans():define how to select
// writetrans():define how to write the new file
// 
// n:No. of photons passing the acrylic sample
// m:No. of photons not passing the acrylic sample
// ne = ( n + m ) / nevent
// nevent = total entries
// thus, ne should be equal to 1, i.e. nevent = n + m
// tran:transmission = n / nevent

scantrees(){
   
   TString name("event_tree");
   FILE* pipe = gSystem->OpenPipe("ls *.root" , "r" );
   
   TString path ;
   TFile* f ;
   TTree* t ;
   
   while( path.Gets(pipe) ){
      cout << path << endl;
      f = TFile::Open( path );
	  t = (TTree*)gROOT->FindObject(name);
	  if( t == NULL ){
         cout << " no object called " << name << " in " << path << endl ; 
	  } else {
		 cout << " found " << name << " in " << path << endl ;
		 //t->Scan("","","",10);
		 calculatetrans(path);
	  }
	  f->Close();
   }
   
   gSystem->Exit( gSystem->ClosePipe( pipe ));
}


void calculatetrans(const char* filepath ){
    
    gSystem->Load("$DYW/InstallArea/$CMTCONFIG/lib/libMCEvent.so");
    dywGLEvent* evt = new dywGLEvent();

    TFile *f = new TFile(filepath);
    if( f->IsZombie() ) {
        cout << "ABORT cannot open " << filepath << endl; 
        return 1 ;
    }
    TTree* t = (TTree*) f->Get("event_tree");
    t->SetBranchAddress("dayabay_MC_event_output",&evt);
    
    Int_t nevent = t->GetEntries();
    Int_t imax = nevent ;
//    imax = 1000;
    
    Int_t nn(0);
    Int_t mm(0);
    Int_t nnn(0);
    Int_t mmm(0);
    for (Int_t i=0;i<imax;i++) {
        t->GetEntry(i);
        if( i%100 == 0 ) cout << i << " "  ; 

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

  
	for(size_t ii=0; ii<nFake; ii++){
	  phd = (dywGLPhotonHitData*)fha[ii];
	  phd->GetPosition(xf,yf,zf);
	  //idf = phd->GetPMTID();
	  //tf  = phd->GetHitTime();
  	  //wf  = phd->GetWeight();
  	}
	
	
	  //start select
//	  if( abs(xf-100.) < 2. ){
          if( -105. < xf && xf < -95.){
		  //cout << "n=" << n << endl;
		  nn++;
		  //cout << "pass_inner= " << nn << endl;
	  }

//          if( abs(xf-110.) < 2. ){
          if( 105. > xf && xf > 95.){
		  //cout << "m=" << m << endl;
                  mm++;
		  //cout << "inner= " << mm << endl;
          }
	  if( -115. < xf && xf < -105.){
		  nnn++;
		  //cout << "pass_outter= " << nnn << endl;
	  }
	  if( 115. > xf && xf > 105.){
		  mmm++;
		  //cout << "outter" << mmm << endl;
	  }
}

 writetrans(filepath,nn,mm,nnn,mmm,nevent,imax);
 delete f;
}

void writetrans(char* filepath, Int_t nn, Int_t mm, Int_t nnn, Int_t mmm, Int_t nevent, Int_t imax){
         Float_t tran,na,ne;

    	TFile *h = new TFile("transmi.root","UPDATE");
        if( h->IsZombie() ) {
	cout << "ABORT cannot open " << filepath << endl;
	return 1 ;
	}


   char dirname[50];

//	cout << filepath << endl;
      sprintf(dirname,"tr%s",filepath);
      TDirectory *cdplane = h->mkdir(dirname);
      cdplane->cd();

         Int_t n,m;
	 n = nn + nnn;
	 m = mm + mmm;

         TNtuple *nt = new TNtuple("fakehits_No_DATA","fakehits_No","pass_inner:pass_outter:inner:outter:total_select_pass:total_d_entries:transmission");
	 Float_t fn,fnevent,fimax;

	 fn = n ;
         fnevent = nevent;
         fimax = imax;
         tran = (Float_t)n/(Float_t)nevent;
         na = n+m;
         Float_t fna;
         fna = na;
         ne = fna/nevent;
	 nt->Fill(nn,nnn,mm,mmm,n,ne,tran);
	 h->Write();
	 delete h;
}
