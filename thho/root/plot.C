//input the x and y asci data file and then plot
#include "Riostream.h"
void plot(void){
	gROOT->Reset();
	c1 = new TCanvas("c1","attenuation v.s. wavelength",200,10,700,500);
	c1->SetFillColor(42);
	c1->SetGrid();

	// draw a frame to define the range
	TH1F *hr = c1->DrawFrame(150,0,800,20);
	hr->SetYTitle("Attenuation m");
	hr->SetXTitle("Wavelength nm");
	c1->GetFrame()->SetFillColor(21);
	c1->GetFrame()->SetBorderSize(12);

	TMultiGraph *mg = new TMultiGraph();

	plot("mineral_wl","mineral_abs","gr1",1,999,1);
	plot("gdls_wlabs","gdls_abs","gr2",2,999,1);
	plot("qe_wl","qe_100","gr3",4,999,70);
	plot("gdls_wlre","gdls_prore","gr4",3,999,10);

	
	
}

Int_t plotmineral(TString filewl, TString fileabs, Int_t lc){

	gROOT->Reset();
	ifstream inmx;
	inmx.open(filewl);
	ifstream inmy;
	inmy.open(fileabs);

	Float_t xx1,yy1;
		
	Int_t i1x = 0;
	
       	const Int_t nm = 1000;
	Double_t xm[nm];
	Double_t ym[nm];
	Double_t exm[nm];
	Double_t eym[nm];
	while (1){
		inmx >> xx1 ;
		if (!inmx.good()) break;
		xm[i1x]=1200/xx1;
		exm[i1x] = 0;
		i1x++;
	}

	Int_t i1y = 0;
	while (1){
		inmy >> yy1;
		if(!inmy.good()) break;
		ym[i1y] = yy1;
		eym[i1y] = 0;
		i1y++;
	}
	
	grm = new TGraphErrors(nm,xm,ym,exm,eym);
	grm->SetLineColor(lc);
//	gr1->SetLineWidth(1);
//	gr1->SetMarkerColor(1);
//	gr1->SetMarkerStyle(21);

	grm->Draw("LP");
	inmx.close();
	inmy.close();

	return 0;
}

Int_t plot(TString filewl, TString fileabs, TString gr, Int_t lc, Int_t size, Int_t we){

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
	
	gr1 = new TGraphErrors(n1,x1,y1,ex1,ey1);
	gr1->SetLineColor(lc);
//	gr->SetLineWidth(1);
//	gr->SetMarkerColor(1);
//	gr->SetMarkerStyle(21);

//	gr->Draw("LP");
	in1x.close();
	in1y.close();

	mg->Add(gr1);
	mg->Draw("ap");
	
	return 0;
}



Int_t plotqe(TString filewl, TString fileabs){

	gROOT->Reset();
	ifstream in2x;
	in2x.open(filewl);
	ifstream in2y;
	in2y.open(fileabs);

	Float_t xx2,yy2;
		
	Int_t i2x = 0;
	
       	const Int_t n2 = 1000;
	Double_t x2[n2];
	Double_t y2[n2];
	Double_t ex2[n2];
	Double_t ey2[n2];
	while (1){
		in2x >> xx2 ;
		if (!in2x.good()) break;
		x2[i2x]=1200/xx2;
		i2x++;
	}

	Int_t i2y = 0;
	while (1){
		in2y >> yy2;
		if(!in2y.good()) break;
		y2[i2y] = yy2*70;
		i2y++;
	}
	
	gr2 = new TGraphErrors(n2,x2,y2,ex2,ey2);
//	gr2->SetLineColor(1);
//	gr2->SetLineWidth(1);
//	gr2->SetMarkerColor(1);
//	gr2->SetMarkerStyle(21);

	gr2->Draw("LP");
	in2x.close();
	in2y.close();

	return 0;
}

