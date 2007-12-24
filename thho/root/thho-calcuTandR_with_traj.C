// use the traj_tree -> optical_phton_final_position to determine
// the total transmittance and
// the total reflection
// 
// Dec.06.2007
// 
// scantrees():scan all trees, select, and write into a new file
// clculatetrans():define how to select
// writetrans():define how to write the new file
//

/*
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
*/

void calculatetrans(const char* filepath ){
    
    gSystem->Load("$DYW/InstallArea/$CMTCONFIG/lib/libMCEvent.so");
    dywGLEvent* evt = new dywGLEvent();

    TFile *f = new TFile(filepath);
    if( f->IsZombie() ) {
        cout << "ABORT cannot open " << filepath << endl; 
        return 1 ;
    }
    TTree* t = (TTree*) f->Get("trajectory_tree");
    t->SetBranchAddress("mc_optical_trajectories",&traj);
    
    Int_t nevent = t->GetEntries();
    Int_t imax = nevent ;
    imax = 1000;
    
    for (Int_t i=0;i<imax;i++) {
        t->GetEntry(i);
        if( i%100 == 0 ) cout << i << " "  ; 

  	//get position, time, weight ,and id data
	Float_t xf(0), yf(0), zf(0);
  	Float_t tf(0) , wf(0) ;
  	Int_t idf(0);

	//phd = (dywGLPhotonHitData*)fha[ii];
	//phd->GetPosition(xf,yf,zf);

	// opa : optical photons traj array
	xf = traj.mFinal.mPosition.fX;
	cout << "x position = " << xf << endl;
	
	//start select
	
}

// writetrans(filepath,nn,mm,nnn,mmm,nevent,imax);
 delete f;
}







/*
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
*/
