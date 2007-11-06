// this code is used to select entries with size of fake hits arrary.
// and then use fAke.code,fAke.weight and fAke.px
//
// first, output the entry with diff constrain with screen? ntuple? histogram?? .....display on screen first
// second, write the random seed and then check the select with visulization
//
// problem
//

//input the file and nFake you want
//void select_diff_nFake_and_other(TString rootfileinput, TString rootfileoutput, int_t nf){
//void select_diff_nFake_and_other(TString rootfileinput, Int_t nf, Int_t codef, Float_t cfw, Float_t cfpx){
void select_diff_nFake_and_other(TString rootfileinput, Int_t nf, Int_t codef){
	dywGLEvent* evt = new dywGLEvent();

	Float_t cfw,cfpx;
	cfw = 0;
	cfpx = 0;
	
	TFile *f = new TFile(rootfileinput,"read");
	//TFile *g = new TFile(rootfileoutput,"RECREATE");
	if( f->IsZombie() ) {
		cout << "ABORT cannot open " << rootfile << endl;
		return 1 ;
	}
	TTree* t = (TTree*) f->Get("event_tree");
	t->SetBranchAddress("dayabay_MC_event_output",&evt);

	Int_t nevent = t->GetEntries();
	Int_t imax = nevent ;
	//imax = 1000;
	

	//hpn ntuplt? histogram??
	Int_t nfn(0),nfnc(0),nfncwapa(0),nfncwapb(0),nfncwbpa(0),nfncwbpb(0);
	
	
	for(Int_t i=0;i<imax;i++){
		t->GetEntry(i);
		if( i%10000 == 0 ) cout << i << endl ;

		//declare and get the fake.code,fAke.weight, and fake.px
		//Int_t codef(0);
		Float_t wf(0);
		Float_t pxf(0),pyf(0),pzf(0);

		TClonesArray &fha = *(evt->GetFakeHitArray());
		const Int_t nFake = fha.GetEntries();
		dywGLPhotonHitData* phd = NULL ;

		//check the special case: nFake =0
		/*Int_t nf0,nfeventID;
		nf0 = 0;
		nfeventID = phd->GetEventID();
		if(nfeventID == 9227){
			cout <<"nFake=" << nFake <<  endl;
		}

		if(nFake == 0){
			nf0++;
			cout << "nf0 = " << nf0 << endl;
		}
		*/


			if(nFake == nf){
				nfn++;
			}
		
		if(nFake !=1 && nFake!=2 && nFake!=3 && nFake!=4 && nFake!=5 ){
			cout << "nFake == 0, " << nFake << endl;
		}




		for(size_t ii=0; ii<nFake; ii++){
				phd = (dywGLPhotonHitData*)fha[ii];
				phd -> GetMomentum(pxf,pyf,pzf);
				wf  = phd->GetWeight();



				
				//codef = phd->GetHitCode();
				//check the value, they should be the same. after check: yes, they are the same.
				
			if( nFake == nf){				
				if(ii == codef){
					nfnc++;
					if(wf > cfw && pxf > cfpx){
						nfncwapa++;
					}
					if(wf > cfw && pxf < cfpx){
						nfncwapb++;
					}
					if(wf < cfw && pxf > cfpx){
						nfncwbpa++;
					}
					if(wf < cfw && pxf < cfpx){
						nfncwbpb++;
					}
				}
				
			}
		}
	}
		    
	cout << "total entries with total fake hits number  " << nf << "  is   " << nfn << endl;
	cout << "total entries with total fake code number  " << codef << "  is   " << nfnc << endl;
	cout << "total entries with  weight > 0 and  px > 0  " <<  "  is   " << nfncwapa << endl;
	cout << "total entries with  weight > 0 and  px < 0  " <<  "  is   " << nfncwapb << endl;
	cout << "total entries with  weight < 0 and  px > 0  " <<  "  is   " << nfncwbpa << endl;
	cout << "total entries with  weight < 0 and  px < 0  " <<  "  is   " << nfncwbpb << endl;
//	cout << "no fake hits number is "<< nf0 << endl;
}
