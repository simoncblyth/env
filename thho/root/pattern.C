//
// Propose:
// Caculate transmittance and reflection of acrylic sample
//
// 

scantrees(){

   TString name("event_tree");
   FILE* pipe = gSystem->OpenPipe("ls *ev.root" , "r" );

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
                 classify_events(path);
          }
          f->Close();
   }

   gSystem->Exit( gSystem->ClosePipe( pipe ));
}



void classify_events(TString rootfileinput ){
	
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
	imax = 100;
	
        TMap* fMap = new TMap ;
	TMap* gMap = new TMap ;
	
	for(Int_t i=0;i<imax;i++){
	   t->GetEntry(i);
	   if( i%10000 == 0 ) cout << "The "<< i << "th event"<< ". All " << imax << " events."<< endl ;
	   
	   TClonesArray &fha = *(evt->GetFakeHitArray());
	   const Int_t nFake = fha.GetEntries();
	  
	   //tag the fake hits pattern per event
	   TString clevt = "" ;
	   for(Int_t ii=0 ; ii<nFake ; ++ii){
	      dywGLPhotonHitData* phd = (dywGLPhotonHitData*)fha[ii];
	      TString clfake = classify_fake(phd );
	      clevt += clfake ;
	      clevt += "," ;
           }	
	   //comment out the below line to speed up selecting
	   //cout << clevt << endl ;

	   //generate different key....that is, cate the events with different way
	   //
	   //make sth. like key "tru,tru," into key "No.1,transmittance"
	   //
	   TString art = classify_pattern(clevt);
	   TString pro = classify_process(art);
	   
	   //countting the diffrent patterns, types, or ways.......
	   counting( fMap, art);
	   counting( gMap, pro);
	}
	
	// dumpping the results cate in different ways.
	creat_asci(rootfileinput);
	dump_map( rootfileinput, gMap,imax );
        dump_map( rootfileinput, fMap,imax );
	close_asci(rootfileinput);
	
//	return fMap;
}

void counting( TMap* map, TString type){
	   TObjString* prev = (TObjString*)map(type.Data()); 
	   if( prev == NULL ){
	       map->Add( new TObjString(type), new TObjString("1"));
	   } else {
	       TString s = prev->GetString();
               Double_t x = atof(s.Data());
               x += 1.0 ;
               TString ns = Form("%f",x);
	       prev->SetString(ns);
	   }   
}


void dump_map( TString rootfilepath, TMap* map, Int_t imax ){
   
   TObjString* s = NULL ;
   TIter next(map);
   Int_t check(0);
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

	Double_t ratio;
	ratio = ((Double_t) vv / (Double_t) imax)*100;
	cout << sk << " " << ratio << " %"<<endl ;
	//output_table(map,sk,ratio);
	check = check + vv;

	output_table(rootfilepath,sk,ratio);
	
   }
   

   if (check == imax){
	   cout << "counting is O.K. No unexpected case" << endl;
	   cout << endl;
	   
   } else{
	   cout << "some unexpected cases may happen!!" << endl;
	   return 1;
   }
   
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


TString classify_pattern( TString clevt ){

	   TString art = "";

	   if( clevt == "" ){
	        cout << "please check the codes" << endl;
	   } else {

		TObjArray * st = clevt.Tokenize(",");
		Int_t nf = st->GetEntries();


		// starting counting. Note not to include the absorption in air.
	
		//reflectance counting
		if( (clevt.BeginsWith("rbo,") == 1) && (clevt.EndsWith("rth,") == 1 && nf == 2)){
			//cout << "reflected!!" << endl;
			//cout << "main/first rebound!!!!!!!!" << endl;
			art = "No.1,reflectance  ";
			//cout << art << endl;
			//cout << endl;
			//gre = gre + vv ;
		} else if( (clevt.BeginsWith("tru,") == 1) && (clevt.EndsWith("rth,") == 1 && nf !=2)){
			//cout << "reflected!!" << endl;
			Int_t onf = ((nf+1)/2);
			//cout << onf << " orders rebound!!!" << endl;
			TString no = Form("No.%i",onf);
			TString label = no;
			label += ",reflectance  ";
			art = label;
			//cout << art << endl;
			//cout << endl;
			//gre = gre + vv ;
		} //transmitance counting and 1 case of aborption
		  else if(clevt.BeginsWith("tru,") == 1 && clevt.EndsWith("tru,") == 1 && nf == 2){
			//cout << nf << "transmittance!!" << endl;
			//cout << "main/first tran through!!!!!" << endl;
			art = "No.1,transmittance";
			//cout << art << endl;
			//cout << endl;
			//gtr = gtr + vv ;
		} else if(clevt.BeginsWith("tru,") == 1 && clevt.EndsWith("tru,") == 1 && nf !=2){
			if(nf == 1){
				//cout << "absorbed in acrylc!!!!" << endl;
				//cout << "absorbed immediately entried acrylic" << endl;
				art = "No.1,absorption   ";
				//cout << art << endl;
				//cout << endl;
				//gab = gab + vv ;
			} else {
				//cout << "transmittance!!" << endl;
				Int_t onf = nf/2;
				//cout << nf/2 << " orders tran through" << endl;
				TString no = Form("No.%i",onf);
				TString label = no;
				label += ",transmittance";
				art = label;
				//cout << art << endl;
				//cout << endl;
				//gtr = gtr + vv ;
			}
		} // absorption counting
		  else if(clevt.BeginsWith("tru,") == 1 && clevt.EndsWith("rbo,") == 1 ){
			//cout << "absorbed in acrylic" << endl;
			//cout << nf/2 << "th rebound and then absorbed in acrylic" << endl;
			Int_t onf = nf/2;
			TString no = Form("No.%i",onf);
			TString label = no;
			label += ",absorption   ";
			art = label;
			//cout << art << endl;
			//cout << endl;
			//gab = gab + vv ;
		} else if(clevt.BeginsWith("tru,") == 1 && clevt.EndsWith("bou,") == 1 ){
			//cout << "absorbed in acrylic!!" << endl;
			//cout << ((nf+1)/2) << "th rebound and then absorbed in acrylic" << endl;
			Int_t onf = ((nf+1)/2);
			TString no = Form("No.%i",onf);
			TString label = no;
			label += ",absorption   ";
			art = label;
			//cout << art << endl;
			//cout << endl;
			//gab = gab + vv ;
		} else {cout << nf << " sth. wrong in your key" << endl; cout << endl;
		}
	
	   }
	//return the pattern types
	return art;

}

TString classify_process(TString art){

	TString type = "";	
	if( art.EndsWith("transmittance")){
		type = "gross transmittance";
	} else if( art.EndsWith("reflectance  ")){
		type = "gross reflectance  ";
	} else if( art.EndsWith("absorption   ")){
		type = "gross absorption   ";
	}

	return type;

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



//  function used to output the T and R result withLaTeX table
//  

void creat_asci(TString rootfile){

	    TString file = rootfile;
	    //file -= ".root";
	    file +=".tex";
	    
	    ofstream o(file.Data());
	    o << Form("\\documentclass{report}\n");
	    o << Form("\\begin{document}\n");
	    o << Form("\\begin{quote}\n");
	    o << Form("\\begin{tabular}{lll}\n");
	    o << Form("\\hline\n");
	    o << Form("$photon energy\$\&\$Process\$\&\$\\\%\$\\\\\n");
	    o << Form("\\hline\n");
  	    o << Form("%s\&\&\\\\\n", file.Data());
    
	    o.close();

	    cout << "creating " << file << " LaTex table sucessfully" << endl;
	    
}

void output_table(TString rootfile, TString sk, Double_t ratio){

	//TString sratio = Form("%f",ratio);
	TString file = rootfile;
	//file -= ".root";
	file +=".tex";
	
	ofstream o(file.Data(),ios::app);
	
	o << Form("\&%s\&%f\\\\\n", sk.Data(), ratio);

	o.close();

	cout << "writing " << file << " " << sk << "Latex table" << endl;
}

void close_asci(TString rootfile){

	    TString file = rootfile;
	    //file -= ".root";
	    file +=".tex";
	    
	    ofstream o(file.Data(),ios::app);
	    
	    o << Form("\\end{tabular}\n");
	    o << Form("\\end{quote}\n");
	    o << Form("\\end{document}\n");

	    o.close();

	    cout << "closing " << file << " LaTex table sucessfully" << endl;
	    
}
