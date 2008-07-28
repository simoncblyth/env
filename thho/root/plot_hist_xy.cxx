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
//			.x main( the x value of x-y)
//			
//			then root will plot input data
//			histogram and x-y graph. y derived
//			from the histogram
//
//			What shold be modified:
//				CHANNEL
//				TGraphErrors *gra = ReadAndPlot("00");
//				void ReadScopeData
//
//////////////////////////////////////////////////////////////

//gROOT->Reset();

// the input data readout channel No.
#define CHANNEL 0
#define PLOTPOINT 7

using namespace std;

struct dataPara {
Int_t iset;
Float_t rms[7];
Float_t mean[7];
};

void ScanData(dataPara *pd){

   FILE* pipe = gSystem->OpenPipe("ls TDC*00" , "r" );

   TString path ;

   Int_t i = 0;
   while( path.Gets(pipe) ){
	  cout << endl << path << endl;
	  ReadData(path,i,pd);
	  i++;
   }

   //gSystem->Exit( gSystem->ClosePipe( pipe ));
}

void plot_hist_xy(void) {

	TCanvas *canvas = new TCanvas("canvas","ccc",200,10,700,500);
	//TH1F *frame = canvas->DrawFrame(0,0,200,3500);
	//frame->SetYTitle("Counts");
	//frame->SetXTitle("ns");

	//TFile* file = new TFile("tmp.root", "recreate");
	TGraphErrors *gra = ReadAndPlot();
	gra->SetLineColor(3);
	gra->SetMarkerStyle(3);
	gra->Draw("AP");
	
}

TGraphErrors *ReadAndPlot(void) {

	dataPara pdata;
	
	ScanData(&pdata);

	Float_t scopeData[PLOTPOINT], scopeDataRMS[PLOTPOINT];
	ReadScopeData(scopeData,scopeDataRMS);
	
	cout << "DDDDDDDDDDDD"<< endl;
	TGraphErrors *gr = new TGraphErrors(7,scopeData,pdata.mean,scopeDataRMS,pdata.rms);
	gr->Fit("pol1");
	cout << "EEEEEEEEEEEEEEEEE"<< endl;
	return gr;

}

void ReadData(TString file, Int_t ifile, dataPara *pdata) {

	TH1F *h = new TH1F("h","channel",3501,0,3500);
	
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
		if(i==CHANNEL) h->Fill(j);
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
	scopeData[0] = 174;
	scopeData[1] = 142;
	scopeData[2] = 126;
	scopeData[3] = 158;
	scopeData[4] = 96;
	scopeData[5] = 64;
	cout << "AAAAAAAAAAAA"<< endl;
	scopeData[6] = 32.8;
	cout << "BBBBBBBBBBB"<< endl;

	for(Int_t i=0;i<7;i++) {
	scopeDataRMS[i] = 0;
	}
	cout << "CCCCCCCCCCCCC"<< endl;
}
