//plot the result of simulation and real data from an asci file

/*
void get_trans_wl(){
   	TFile *f = new TFile("transmi.root");
	TDirectory *cdtrans = f->Cd("trasmi.root:/2.1ev
   	TTree *ntuple = (TTree*)f->Get("fakehis_No_DATA");
   	TFile *g = new TFile("transmiplotdata.root","RECREATE");
   	Float_t w,t;
   	ntuple->SetBranchAddress("tnsmission",&t);
   	Int_t nentries = (Int_t)ntuple->GetEntries();
	for (Int_t i=0; i<nentries; i++){
	   ntuple->GetEntry(i);
	   TNtuple *nt = new TNtuple("transmission_wavelength","transmission_wavelength",
	}

*/

{
	gROOT->Reset();
#include "Riostream.h"
	c1 = new TCanvas("c1","transmission v.s. wavelength",200,10,700,500);
	c1->SetFillColor(42);
	c1->SetGrid();

	// draw a frame to define the range
	TH1F *hr = c1->DrawFrame(150,0,650,1);
	hr->SetYTitle("Transmission % ");
	hr->SetXTitle("Wavelenth nm");
	c1->GetFrame()->SetFillColor(21);
	c1->GetFrame()->SetBorderSize(12);
	
	Int_t n1 = 45;
	Double_t x1[]={620.000000,590.476190,563.636364,539.130435,516.666667,496.000000,476.923077,459.259259,442.857143,427.586207,413.333333,400.000000,387.500000,375.757576,364.705882,354.285714,344.444444,335.135135,326.315789,317.948718,310.000000,302.439024,295.238095,288.372093,281.818182,275.555556,269.565217,263.829787,258.333333,253.061224,248.000000,243.137255,238.461538,233.962264,229.629630,225.454545,221.428571,217.543860,213.793103,210.169492,206.666667,203.278689,200.000000,196.825397,193.750000};
	Double_t y1[]={0.8907,0.8905,0.8903,0.89,0.8897,0.8895,0.8892,0.889,0.8887,0.8883,0.8878,0.8872,0.8867,0.8862,0.8856,0.885,0.8844,0.8838,0.883,0.8822,0.8817,0.881,0.8793,0.8766,0.8736,0.8706,0.867,0.8632,0.8582,0.8533,0.8482,0.8419,0.835,0.8268,0.8165,0.8041,0.7885,0.7668,0.7359,0.6877,0.6018,0.4045,0,0,0};
	Double_t ex1[n1];
	Double_t ey1[n1];
	for(Int_t i=0;i<n1;i++){
		ey1[i] = (sqrt(y1[i]*100000))/100000;
		//ex[i] = (sqrt(y[i]))+10;
	}
	gr1 = new TGraphErrors(n1,x1,y1,ex1,ey1);
	//gr = new TGraph(n,x,y);
 	gr1->SetLineColor(2);
      	gr1->SetLineWidth(1);
        gr1->SetMarkerColor(4);
	gr1->SetMarkerStyle(21);
   	//gr1->SetTitle("a simple graph");
	//gr1->GetXaxis()->SetTitle("Wavelength");
	//gr1->GetYaxis()->SetTitle("Transmission");
	gr1->Draw("LP");


	ifstream in;
	in.open("ntu-a1.asc");
	Float_t xx,yy;
	Int_t j = 0;
	Int_t n2 = 401;
	Double_t x2[n2];
	Double_t y2[n2];
	Double_t ex2[n2];
	Double_t ey2[n2];
	while (1){
		in >> xx >> yy;
		if (!in.good()) break;
		x2[j]=xx;
		y2[j]=yy*0.01;
		j++;
	}
	gr2 = new TGraphErrors(n2,x2,y2,ex2,ey2);
	//gr = new TGraph(n,x,y);
 	//gr2->SetLineColor(3);
      	//gr2->SetLineWidth(1);
        //gr2->SetMarkerColor(4);
	//gr2->SetMarkerStyle(21);
   	//gr2->SetTitle("a simple graph");
	//gr2->GetXaxis()->SetTitle("Wavelength");
	//gr2->GetYaxis()->SetTitle("Transmission");
	gr2->Draw("LP");

	//c1->Update();
	//c1->GetFrame()->SetFillColor(21);
	//c1->GetFrame()->SetBorderSize(12);
	//c1->Modified();
}
	

