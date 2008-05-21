// plotting the deviation of two values
//
//
//
//
#include "Riostream.h"

TString datafiledoc = "dataresult";
Float_t tulim(0), tblim(0);

void main(void){
//	gROOT->Reset();

	c1 = new TCanvas("c1","c1",200,10,700,500);
        TH1F* hr1 = c1->DrawFrame(0,-0.1,850,0.5);//the xy axis the fram (xmin,ymin,xmax,ymax)
	TH1F* hr1 = run(700,hr1);

	cout << "completing plotting" << endl;
}
TH1F* run(Double_t wl, TH1F* hr){



	TString ai = "a4std";
	TString af = "a5std";
	
	TString tit = " delta transmittance v.s. wavelength";
	hr->SetTitle(tit);
        hr->SetYTitle("delta transmittance % ");
        hr->SetXTitle("wavelength nm ");

	//modify the input file name and
	//the setting or profile of the plotting
	/*TGraphErrors* gr1 = plot(ai,af,wl);
        gr1->SetLineColor(1);
        gr1->SetMarkerColor(1);
        gr1->SetMarkerStyle(1);
        gr1->Draw("P");
*/
	TGraphErrors* gr2 = plot(ai,af,wl-50);
        gr2->SetLineColor(1);
        gr2->SetMarkerColor(1);
        gr2->SetMarkerStyle(1);
        gr2->Draw("P");

	TGraphErrors* gr3 = plot(ai,af,wl-100);
	grdraw(gr3);

	TGraphErrors* gr3 = plot(ai,af,wl-150);
	grdraw(gr3);

	TGraphErrors* gr4 = plot(ai,af,wl-200);
	grdraw(gr4);

	TGraphErrors* gr5 = plot(ai,af,wl-250);
	grdraw(gr5);

	TGraphErrors* gr6 = plot(ai,af,wl-300);
	grdraw(gr6);

	TGraphErrors* gr7 = plot(ai,af,wl-350);
	grdraw(gr7);

	TGraphErrors* gr8 = plot(ai,af,wl-400);
	grdraw(gr8);
	
	return hr;
	delete hr;
	cout << "completing running" << endl;
}

void grdraw(TGraphErrors* gr){
	gr->SetLineColor(1);
	gr->SetMarkerColor(1);
	gr->SetMarkerStyle(1);
	gr->Draw("P");
}

TGraphErrors* plot(TString filea, TString fileb, Double_t wl){
	
	Double_t* fa;
	Double_t* fb;
//	cout << "fa add " << &fa << " fb add " << &fb << endl;
//	cout << "real a add " << *read(filea,wl) << endl;
	fa = read(filea,wl);
	fb = read(fileb,wl);
//	cout << "fa add " << &fa << " fb add " << &fb << endl;
//	cout << " fa[0] " << fa[0] << " fa[1] " << fa[1] << endl;
//	cout << " fb[0] " << fb[0] << " fb[1] " << fb[1] << endl;

	
	Double_t yy(0),yye(0);

	yy = fb[0]-fa[0];
	yye = sqrt(fb[1]*fb[1]+fa[1]*fa[1]);
	
	Double_t wla[1] = {wl};
	Double_t wlae[1] = {0};
	Double_t ry[1] = {yy};
	Double_t rye[1] = {yye};

	/*
	cout << "wave length " << wla[0] << "      wavelength error  " << wlae[0] << endl;
	cout << "delta value " << ry[0]  << "   delta value error " << rye[0] << endl;
	cout << "value a     " << fa[0]  << "   error of a        " << fa[1] << endl;
	cout << "value b     " << fb[0]  << "   error of b        " << fb[1] << endl;
	cout << endl;
	*/

	cout << wla[0] << " " << ry[0] << " " << rye[0] << endl;

	
	//plotting
	gr = new TGraphErrors(1,wla,ry,wlae,rye);
        //gr = new TGraph(n,x,y);

	return gr;
	delete gr;
	
}

Double_t* read(TString file, Double_t wl){

	Int_t m = size(file);	
	const Int_t n = m;
	Double_t ax[n],ay[n],awl[n];
	Double_t* r = new Double_t[n];
	
	Int_t i(0);
	Double_t x(0),y(0);
	ifstream in;
	in.open(file);
	while(1){
		in >> x >> y;
		if(!in.good()) break;
		ax[i] = x;
		ay[i] = y;
		awl[i] = 200 + i;
		if(awl[i] == wl){
			r[0] = ax[i];
			r[1] = ay[i];
		}
	
		i++;
	}
	in.close();

//	cout << "x " << r[0] << " " << "y " << r[1] << endl;
//	cout << "r " << r << endl;
	return r;
	delete [ ] r;
}

Int_t size(TString file){

	Int_t m(0);
	ifstream im;
	//decide the size of the matrix, don't modify it
	im.open(file);
	Double_t dxx(0),dyy(0);
	//cout << "start counting data size"<< endl;
	while(1){
		im >> dxx >> dyy;
		if(!im.good()) break;
		m++;
		//cout << m << " " << endl;
	}
	im.close();

	return m;
}
