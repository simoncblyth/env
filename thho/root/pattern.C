//
// Propose:
// Caculate transmittance and reflection of acrylic sample
//
// 
TMap* classify_events(TString rootfileinput ){
	
	cout << "selecting....." << endl;
	dywGLEvent* evt = new dywGLEvent();
	TFile *f = new TFile(rootfileinput,"read");
	if( f->IsZombie() ) {
		cout << "ABORT cannot open " << rootfile << endl;
		return 1 ;
	}
	TTree* t = (TTree*) f->Get("event_tree");
	t->SetBranchAddress("dayabay_MC_event_output",&evt);
	
	Int_t nevent = t->GetEntries();
	Int_t imax = nevent ;
	// constrain the No. of events to loop in order to save time when testing
	//imax = 1000;
	
        TMap* fMap = new TMap ;
	
	for(Int_t i=0;i<imax;i++){
	   t->GetEntry(i);
	   if( i%10000 == 0 ) cout << "The "<< i << "th event"<< ". All " << imax << " events."<< endl ;
	   
	   TClonesArray &fha = *(evt->GetFakeHitArray());
	   const Int_t nFake = fha.GetEntries();
	   
	   TString clevt = "" ;
	   for(Int_t ii=0 ; ii<nFake ; ++ii){
	      dywGLPhotonHitData* phd = (dywGLPhotonHitData*)fha[ii];
	      TString clfake = classify_fake(phd );
	      clevt += clfake ;
	      clevt += "," ;
           }	
	   //cout << clevt << endl ;

	   TObjString* prev = (TObjString*)fMap(clevt.Data()); 
	   if( prev == NULL ){
	       fMap->Add( new TObjString(clevt), new TObjString("1"));
	   } else {
	       TString s = prev->GetString();
               Double_t x = atof(s.Data());
               x += 1.0 ;
               TString ns = Form("%f",x);
	       prev->SetString(ns);
	   }   
	}
	
        dump_map( fMap );
	
	return fMap;
}

void dump_map( TMap* map){

   //gross reflectance and transmittance counter
   Double_t gre(0);
   Double_t gtr(0);
   Double_t gab(0);

   TObjString* s = NULL ;
   TIter next(map);
   while( (s = (TObjString*)next()) ){
	TString sk = s->GetString();
        TObjString* v = ((TObjString*)map(sk.Data()))	;

	Double_t vv(-1); 
	if( v == NULL ){
	   cout << "oops null value " << endl ;
	   return ;
	} else {
	   TString sv = v->GetString();
	   vv = atof(sv.Data());
	}
        // EndsWith  BeginsWith

	cout << " key " << sk << " " << vv << endl ;

	TObjArray * st = sk.Tokenize(",");
	Int_t nf = st->GetEntries();

		// starting counting. Note not to include the absorption in air.
	
		//reflectance counting
		if( (sk.BeginsWith("rbo,") == 1) && (sk.EndsWith("rth,") == 1 && nf == 2)){
			cout << "reflected!!" << endl;
			cout << "main/first rebound!!!!!!!!" << endl;
			cout << endl;
			gre = gre + vv ;
		} else if( (sk.BeginsWith("tru,") == 1) && (sk.EndsWith("rth,") == 1 && nf !=2)){
			cout << "reflected!!" << endl;
			cout << ((nf+1)/2) << " orders rebound!!!" << endl;
			cout << endl;
			gre = gre + vv ;
		} //transmitance counting and 1 case of aborption
		  else if(sk.BeginsWith("tru,") == 1 && sk.EndsWith("tru,") == 1 && nf == 2){
			cout << nf << "transmittance!!" << endl;
			cout << "main/first tran through!!!!!" << endl;
			cout << endl;
			gtr = gtr + vv ;
		} else if(sk.BeginsWith("tru,") == 1 && sk.EndsWith("tru,") == 1 && nf !=2){
			if(nf == 1){
				cout << "absorbed in acrylc!!!!" << endl;
				cout << "absorbed immediately entried acrylic" << endl;
				cout << endl;
				gab = gab + vv ;
			} else {
				cout << "transmittance!!" << endl;
				cout << nf/2 << " orders tran through" << endl;
				cout << endl;
				gtr = gtr + vv ;
			}
		} // absorption counting
		  else if(sk.BeginsWith("tru,") == 1 && sk.EndsWith("rbo,") == 1 ){
			cout << "absorbed in acrylic" << endl;
			cout << nf/2 << "th rebound and then absorbed in acrylic" << endl;
			cout << endl;
			gab = gab + vv ;
		} else if(sk.BeginsWith("tru,") == 1 && sk.EndsWith("bou,") == 1 ){
			cout << "absorbed in acrylic!!" << endl;
			cout << ((nf+1)/2) << "th rebound and then absorbed in acrylic" << endl;
			cout << endl;
			gab = gab + vv ;
		} else {cout << nf << " sth. wrong in your key" << endl; cout << endl;
		}
	
   }
   cout << "gross reflectance is   " << gre << endl;
   cout << "gross transmittance is " << gtr << endl;
   cout << "absorbed in acrylic    " << gab << endl;
}



TString classify_fake( void* pphd ){
	
	Float_t wf(0);
	Float_t pxf(0),pyf(0),pzf(0);

        Float_t cfw(0.);
        Float_t cfpx(0.);

	dywGLPhotonHitData* phd = (dywGLPhotonHitData*)pphd ;
	phd -> GetMomentum(pxf,pyf,pzf);
	wf  = phd->GetWeight();
	//codef = phd->GetHitCode();
	//check the value, they should be the same. after check: yes, they are the same.

        TString s = "" ;	
	if(wf > cfw && pxf > cfpx){   s = "rbo" ;
	} else if(wf > cfw && pxf < cfpx){ s = "bou" ;
	} else if(wf < cfw && pxf > cfpx){ s= "rth";
	} else if(wf < cfw && pxf < cfpx){ s= "tru";
	}
	return s;
}
	

// function defined how to save the result per selection with an nFake and an fAke.code
//
/*
Int_t save_selection_nf_zero(TString rootfileinput, Int_t nfn) {

	TFile * h = new TFile("select.root","UPDATE");
	if( h->IsZombie() ) {
		cout << "ABORT cannot open " << rootfileoutput << endl;
		return 1 ;
	}

	char dirname[50];

	sprintf(dirname,"select%swith%iwithnohit",rootfileinput,nfn);
	TDirectory *cdplane = h->mkdir(dirname);
	cdplane->cd();
	TNtuple *nt = new TNtuple("fakehits_select","fakehits_select","total");
	nt->Fill(nfn);
	h->Write();
	delete h;

	return 1;
	
}
Int_t save_selection(TString rootfileinput, Int_t nf, Int_t codef, Int_t nfn, Int_t nfnc, Int_t nfncwapa, Int_t nfncwapb, Int_t nfncwbpa, Int_t nfncwbpb){

	TFile * h = new TFile("select.root","UPDATE");
	if( h->IsZombie() ) {
		cout << "ABORT cannot open " << rootfileoutput << endl;
		return 1 ;
	}

	char dirname[50];

	sprintf(dirname,"select%swith%i%i",rootfileinput,nf,codef);
	TDirectory *cdplane = h->mkdir(dirname);
	cdplane->cd();
	TNtuple *nt = new TNtuple("fakehits_select","fakehits_select","total:code:wapa:wapb:wbpa:wbpb");
	nt->Fill(nfn,nfnc,nfncwapa,nfncwapb,nfncwbpa,nfncwbpb);
	h->Write();
	delete h;

	return 1;
	
}
*/




// function defined how to calculate T and R.
Int_t calcuTandR_write(TString outputfile){
	
}

//  function used to output the T and R result withLaTeX table
//  


Int_t output_table(TString pattern, Float_t counts){
	
}
