//input the x and y asci data file and then plot
#include "Riostream.h"
void plot(void){
	gROOT->Reset();
	c1 = new TCanvas("c1","attenuation v.s. wavelength",200,10,700,500);
	c1->SetFillColor(42);
	c1->SetGrid();

	// draw a frame to define the range
//	TH1F *hr = c1->DrawFrame(150,0,800,20);
//	hr->SetYTitle("Attenuation m");
//	hr->SetXTitle("Wavelength nm");
	c1->GetFrame()->SetFillColor(21);
	c1->GetFrame()->SetBorderSize(12);

	TMultiGraph *mg = new TMultiGraph();
	
	TGraphErrors *gr1;
	TGraphErrors *g1 = plotg("mineral_wl","mineral_abs",gr1,1,999,1);
	mg->Add(g1);

	TGraphErrors *gr2;
	TGraphErrors *g2 = plotg("gdls_wlabs","gdls_abs",gr2,2,999,1);
	mg->Add(g2);
	TGraphErrors *gr3;
        TGraphErrors *g3 = plotg("qe_wl","qe_100",gr3,4,999,70);
	mg->Add(g3);
	//TGraphErrors *gr4 = plotg("gdls_wlre","gdls_prore",gr4,3,999,10);
	//mg->Add(gr4);

	c1->Update();
	c1->Modified();
	mg->Draw("ap");
	
}

TGraphErrors* plotg(TString filewl, TString fileabs, TGraphErrors *gr, Int_t lc, Int_t size, Int_t we){

	gROOT->Reset();
	ifstream in1x;
	in1x.open(filewl);
	ifstream in1y;
	in1y.open(fileabs);

	Float_t xx1,yy1;
		
	Int_t i1x = 0;
	
       	const Int_t n1 = size;
	Double_t x1[n1];
	Double_t y1[n1];
	Double_t ex1[n1];
	Double_t ey1[n1];
	while (1){
		in1x >> xx1 ;
		if (!in1x.good()) break;
		x1[i1x]=1200/xx1;
		ex1[i1x] = 0;
		i1x++;
	}

	Int_t i1y = 0;
	while (1){
		in1y >> yy1;
		if(!in1y.good()) break;
		y1[i1y] = yy1*we;
		ey1[i1y] = 0;
		i1y++;
	}
	
	gr = new TGraphErrors(n1,x1,y1,ex1,ey1);
	gr->SetLineColor(lc);
//	gr->SetLineWidth(1);
//	gr->SetMarkerColor(1);
//	gr->SetMarkerStyle(21);

//	gr->Draw("LP");
	in1x.close();
	in1y.close();

	return gr;
	
}
