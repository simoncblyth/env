//////////////////////////////////////////////////////////
//
//	Filename:	plot_hist_xy_diff_fsr.cxx
//	Created by:	Taihsiang Ho
//	Date:		2008, Aug, 18
//	Description:	A root macro to input a columu data,
//			make a histogram, and use the histogram
//			to plot the x-y
//
//	Usage: 		input data:	channel counts
//					FSR	40
//					0	444
//					0	983
//					0	667
//					FSR	70
//					0	222
//					1	231
//					
//			> root
//			.L plot_hist_xy_diff_fsr.C
//			plot_hist_xy_diff_fsr.C( the channel value)
//			
//			then root will plot input data
//			histogram and x-y graph by scanning the dir.
//			 y derived from the histogram and x are the given FSR reg setting
//
//			 once the channel value is what you want,
//			 LSB would be y value.
//
//			 What shold be modified:
//				PLOTPOINT  (how many data points?)
//				jj = (148/jj); // the constant depend on the input signal
//				jj would be the LSB, or say, the resolution
//
//////////////////////////////////////////////////////////////

//gROOT->Reset();

// the input data readout channel No.
#define PLOTPOINT 14

using namespace std;

struct dataPara {
Float_t fsr[PLOTPOINT];
Float_t fsrrms[PLOTPOINT];
Float_t rms[PLOTPOINT];
Float_t mean[PLOTPOINT];
};

void ScanData(dataPara *pd, Int_t channelcheck){

   FILE* pipe = gSystem->OpenPipe("ls 000*" , "r" );

   TString path ;

   Int_t i = 0;
   while( path.Gets(pipe) ){
	  cout << endl << path << endl;
	  ReadData(path,pd,channelcheck);
	  i++;
   }

   //gSystem->Exit( gSystem->ClosePipe( pipe ));
}

void plot_hist_xy_diff_fsr(Int_t channelcheck) {

	TCanvas *canvas = new TCanvas("fsrchannel","LSB v.s. Full Scale Range Register setting",200,10,700,500);
	TH1F *frame = canvas->DrawFrame(0,0,200,0.6);
	frame->SetYTitle("LSB");
	frame->SetXTitle("Full Scale Range Register setting");

	gStyle->SetOptFit(0111);
	//TFile* file = new TFile("tmp.root", "recreate");
	ReadAndPlot(channelcheck);


	
}

void ReadAndPlot(Int_t channelcheck) {

	dataPara pdata;
	
	ScanData(&pdata, channelcheck);

	
	TGraphErrors *gr = new TGraphErrors(PLOTPOINT,pdata.fsr,pdata.mean,pdata.fsrrms,pdata.rms);
	TF1 *func = new TF1("fit",fitf,0,200,1);
	func->SetParameter(0,9);
	func->SetParNames("Constant");
	gr->Fit("fit");
	gr->SetLineColor(1);
	gr->SetMarkerColor(3);
	gr->SetMarkerStyle(3);
	gr->Draw("P");
	//return gr;

}

void ReadData(TString file, dataPara *pdata, Int_t channelcheck) {

	TH1F *h = new TH1F("h","channel",4001,0,4000);	
	
	ifstream inputSizeFile,inputDataFile;
	inputSizeFile.open(file);

	Int_t inputSizeCount(0);
	while(1) {
		TString i,j;
		inputSizeFile >> i >> j;
		if(!inputSizeFile.good()) break;
		inputSizeCount++;
	}
	inputSizeFile.close();

	const Int_t inputSize = inputSizeCount;

	cout << "size is " << inputSizeCount << endl;
	cout << "size is " << inputSize << endl;
	
	inputDataFile.open(file);
	Int_t fsrArrayIndex(0);

	Initialization(pdata);

	Int_t cc(0);
	while(1) {
		TString i,j;
		inputDataFile >> i >> j;
		//cout << " i and j are " << i << " and " << j << endl;
		if(i=="FSR") {
			if(pdata->fsr[fsrArrayIndex] != 0) {
				//cout << "DDDDDDDDDDD"<< endl;
				pdata->mean[fsrArrayIndex] = h->GetMean();
				cout << "mean " << pdata->mean[fsrArrayIndex] << endl;
				pdata->rms[fsrArrayIndex] = h->GetRMS();
				cout << "RMS  " << pdata->rms[fsrArrayIndex] << endl;
				fsrArrayIndex++;
				if(fsrArrayIndex<PLOTPOINT) {
					cout << "FSR value is " << j << endl;
					Float_t jj = atof(j.Data());
					pdata->fsr[fsrArrayIndex] = jj;
					h->Reset();
				}
				//cout << "AAAAAAAAAAA"<< endl;
			} else {
				Float_t jj = atof(j.Data());
				pdata->fsr[fsrArrayIndex] = jj;
				//cout << "BBBBBBBBBBB" << endl;
				cout << "the read in FSR points for jj are " << fsrArrayIndex << " and " << jj << endl;
			}
		} else if(!inputDataFile.good()) {
			pdata->mean[fsrArrayIndex] = h->GetMean();
                        cout << "mean " << pdata->mean[fsrArrayIndex] << endl;
                        pdata->rms[fsrArrayIndex] = h->GetRMS();
                        cout << "RMS  " << pdata->rms[fsrArrayIndex] << endl;
			break;
		} else {
			//cout << "CCCCCCCCCCCCCC" << endl;
			//cout << endl << pdata->fsr[fsrArrayIndex]<< endl;
			Float_t ii, jj;
			ii = atof(i.Data());
			jj = atof(j.Data());
			jj = (148/jj); // the constant depend on the input signal
			if(ii==channelcheck) h->Fill(jj);
		}
	}
	inputDataFile.close();

	cout << "the read in FSR points are~~~ " << fsrArrayIndex+1 << endl;
	//check the read in data correct or not
	if(fsrArrayIndex==(PLOTPOINT-1)) cout << "the read in FSR points are " << fsrArrayIndex+1 << endl;
	else cout << "sth. wrong with your reading data process " << endl;

	delete h;

}

void Initialization(dataPara* pdata) {

	for(Int_t i=0;i<PLOTPOINT;i++) {
		pdata->fsr[i]=0;
		pdata->fsrrms[i] = 0;
	}
}

Double_t fitf(Double_t *x, Double_t *par) {

	Double_t fitval = par[0]/x[0];

	return fitval;

}
