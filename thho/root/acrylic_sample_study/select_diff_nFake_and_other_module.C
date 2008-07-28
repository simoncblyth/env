//
// Propose:
// Caculate transmittance and reflection of acrylic sample
// 
// This macro is used to select entries with size of fake hits arrary,
// and then use fAke.code,fAke.weight and fAke.px
// Finally use the information to calculate the (total/gross passing) transmittance and reflection
//
//
// first, output the entry with diff constrain with screen? ntuple? histogram?? .....display on screen first
// second, write the random seed and then check the select with visulization
//
// problem
//
// Process:
// Nov.6 generalize the codes plot histogram instead of ntuple
// Nov.6 generalize the codes to select automatically
// Nov.8 starting modulize the code for easy to read
// Nov.8 caculate T and R
//
//
//Dec.16 begin use histogram instead of counting

//input the file and nFake you want
//void select_diff_nFake_and_other(TString rootfileinput, TString rootfileoutput, int_t nf){
//void select_diff_nFake_and_other(TString rootfileinput, Int_t nf, Int_t codef, Float_t cfw, Float_t cfpx){
void select_diff_nFake_and_other_module(TString rootfileinput){
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
	imax = 1000;
	


	//create an array to reserve different nFake number
	//show the max fake hit number per event in all data
	Int_t countnfi(0);
	cout << "finding max fake hit number in one event and creanting array..." << endl ;
	for(Int_t j=0;j<imax;j++){
		t->GetEntry(j);
		if( j%10000 == 0 ) {
			cout << "The "<< j << "th event"<< ". All " << imax << " events."<< endl ;
		}

		TClonesArray &fha = *(evt->GetFakeHitArray());
		const Int_t nFake = fha.GetEntries();

		if(nFake > countnfi){
			countnfi = nFake;	
		}
		
		
	}
	cout << " the max No. of fake hit array is " << countnfi << endl;
	const Int_t countnfic = countnfi;
	Int_t countnfiar[countnfic];

	//starting selecting
	select(rootfileinput,countnfic,imax);
}


//function defined what to be select
Int_t select(TString rootfileinput,Int_t countnfic,Int_t imax){

	cout << "selecting....." << endl;

	dywGLEvent* evt = new dywGLEvent();
	
	TFile *f = new TFile(rootfileinput,"read");
	if( f->IsZombie() ) {
		cout << "ABORT cannot open " << rootfile << endl;
		return 1 ;
	}
	TTree* t = (TTree*) f->Get("event_tree");
	t->SetBranchAddress("dayabay_MC_event_output",&evt);
	
	Float_t cfw,cfpx;
	cfw = 0;
	cfpx = 0;

	// creat the histogram in order to record the select entries associated diff fake hits number etc.
	hp0 = new TH1F("hpx","This is the px distribution",100,-4,4);


	for(Int_t nf=0;nf<=countnfic;nf++){
		
		// the size of fake array equal to 0 is a special value that codef loop would not work
		if(nf == 0){
			Int_t nfn0(0);
			for(Int_t i=0;i<imax;i++){
				
				t->GetEntry(i);
				if( i%10000 == 0 ) {
					cout << "The "<< i << "th event"<< ". All " << imax << " events."<< endl ;
				}
				TClonesArray &fha = *(evt->GetFakeHitArray());
				const Int_t nFake = fha.GetEntries();
				dywGLPhotonHitData* phd = NULL ;

				if(nFake == nf){
					nfn0++;
				}
			}
			cout << "no fake hits No. is " << nfn0 << endl;
			save_selection_nf_zero(rootfileinput,nfn0);
		}

		
		for(Int_t codef=0;codef<nf;codef++){
			
			Int_t nfn(0),nfnc(0),nfncwapa(0),nfncwapb(0),nfncwbpa(0),nfncwbpb(0);

			for(Int_t i=0;i<imax;i++){
				t->GetEntry(i);

				//show the selecting process				
				if( i%10000 == 0 ) {
				cout << "The "<< i << "th event"<< ". All " << imax << " events."<< endl ;
				}

				//declare and get physics quantity the fake.code,fAke.weight, and fake.px
				Float_t wf(0);
				Float_t pxf(0),pyf(0),pzf(0);

				TClonesArray &fha = *(evt->GetFakeHitArray());
				const Int_t nFake = fha.GetEntries();
				dywGLPhotonHitData* phd = NULL ;
		
				if(nFake == nf){
					nfn++;
				}
		
				for(size_t ii=0; ii<nFake; ii++){
			
					if( nFake == nf){
						
						if(ii == codef){
							nfnc++;
							//cout << nfnc << endl;
							
							phd = (dywGLPhotonHitData*)fha[ii];
							phd -> GetMomentum(pxf,pyf,pzf);
							wf  = phd->GetWeight();
							//codef = phd->GetHitCode();
							//check the value, they should be the same. after check: yes, they are the same.
	
							if(wf > cfw && pxf > cfpx){
								nfncwapa++;
							}
							else if(wf > cfw && pxf < cfpx){
								nfncwapb++;
							}
							else if(wf < cfw && pxf > cfpx){
								nfncwbpa++;
							}
							else if(wf < cfw && pxf < cfpx){
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
        cout << rootfileinput << endl;
	
	if( nf != 0 ) {
	save_selection(rootfileinput,nf,codef,nfn,nfnc,nfncwapa,nfncwapb,nfncwbpa,nfncwbpb);
	}
	
		}
	}
	return 0;
}

// function defined how to save the result per selection with an nFake and an fAke.code
//

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





// function defined how to calculate T and R.
// (and then plot) and write into an file.
Int_t calcuTandR_write(TString outputfile){
	
}


