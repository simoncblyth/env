//////////////////////////////////////////////////////////
//
//	Filename:	plot_hist_xy.cxx
//	Created by:	Taihsiang Ho
//	Date:		2008, July, 24
//	Description:	A root macro to input a columu data,
//			make a histogram, and use the histogram
//			to plot the x-y
//
//	Usage: 		input data:	channel counts
//					0	444
//					0	983
//					0	667
//					
//			> root
//			.L plot_hist_x_y.C
//			plot_hist_x_y.C( the channel value)
//			
//
//			then root will plot input data
//			histogram and x-y graph by scanning the dir.
//			 y derived from the histogram
//
//			 once the channel value is what you want,
//			 counts would be y value.
//			 specify x value with modifying ReadScopeData
//
//			What shold be modified:
//				CHANNEL
//				PLOTPOINT
//				TGraphErrors *gra = ReadAndPlot("00");
//				void ReadScopeData
//
//////////////////////////////////////////////////////////////

//gROOT->Reset();

// the input data readout channel No.
//#define CHANNEL 0
#define PLOTPOINT 6

using namespace std;

struct dataPara {
Int_t iset;
Float_t rms[PLOTPOINT];
Float_t mean[PLOTPOINT];
};

void ScanData(dataPara *pd, Int_t channelcheck){

   FILE* pipe = gSystem->OpenPipe("ls TDC*" , "r" );

   TString path ;

   Int_t i = 0;
   while( path.Gets(pipe) ){
	  cout << endl << path << endl;
	  ReadData(path,i,pd,channelcheck);
	  i++;
   }

   //gSystem->Exit( gSystem->ClosePipe( pipe ));
}

void plot_hist_xy(Int_t channelcheck) {

	TCanvas *canvas = new TCanvas("canvas","counts v.s. input",200,10,700,500);
	TH1F *frame = canvas->DrawFrame(0,0,500,4500);
	frame->SetYTitle("Counts");
	frame->SetXTitle("ns");

	gStyle->SetOptFit(0111);
	//TFile* file = new TFile("tmp.root", "recreate");
	ReadAndPlot(channelcheck);


	
}

void ReadAndPlot(Int_t channelcheck) {

	dataPara pdata;
	
	ScanData(&pdata, channelcheck);

	Float_t scopeData[PLOTPOINT], scopeDataRMS[PLOTPOINT];
	ReadScopeData(scopeData,scopeDataRMS);
	
	TGraphErrors *gr = new TGraphErrors(PLOTPOINT,scopeData,pdata.mean,scopeDataRMS,pdata.rms);
	gr->Fit("pol1");
	gr->SetLineColor(1);
	gr->SetMarkerColor(3);
	gr->SetMarkerStyle(3);
	gr->Draw("P");
	//return gr;

}

void ReadData(TString file, Int_t ifile, dataPara *pdata, Int_t channelcheck) {

	TH1F *h = new TH1F("h","channel",4001,0,4000);
	
	ifstream inputSizeFile,inputDataFile;
	inputSizeFile.open(file);

	Int_t inputSizeCount(0);
	while(1) {
		Float_t i,j;
		inputSizeFile >> i >> j;
		if(!inputSizeFile.good()) break;
		inputSizeCount++;
	}
	inputSizeFile.close();

	const Int_t inputSize = inputSizeCount;

	cout << "size is " << inputSizeCount << endl;
	cout << "size is " << inputSize << endl;
	
	inputDataFile.open(file);
	while(1) {
		Float_t i,j;
		inputDataFile >> i >> j;
		if(i==channelcheck) h->Fill(j);
		if(!inputDataFile.good()) break;
	}
	inputDataFile.close();
	
	pdata->iset = ifile;
	pdata->mean[ifile] = h->GetMean();
	cout << "mean " << pdata->mean[ifile] << endl;
	pdata->rms[ifile] = h->GetRMS();
	cout << "RMS  " << pdata->rms[ifile] << endl;

	delete h;
	
}

void ReadScopeData(Float_t scopeData[], Float_t scopeDataRMS[]) {	
	scopeData[0] = 250;
	scopeData[1] = 300;
	scopeData[2] = 350;
	scopeData[3] = 400;
	scopeData[4] = 450;
	scopeData[5] = 200;

	Int_t ini = PLOTPOINT;
	
	for(Int_t i=0;i<ini;i++) {
	scopeDataRMS[i] = 0;
	}
}
