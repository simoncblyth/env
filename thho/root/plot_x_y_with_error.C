// root macro for plotting x-y
// the code will read an asii file "test.asc" and then plotting
// the fromat of test.asc should be
//
// x y
// 1 2
// 3 4
// 3 3
// 
// x:(average) value
// y:error bar
//
//
//
#include "Riostream.h"

TString datafiledoc = "dataresult";
Float_t tulim(0), tblim(0);

void main(void){
//	gROOT->Reset();

//	create_tex(datafiledoc);

	interval(89.);
	c1 = new TCanvas("c1","c1",200,10,700,500);
        TH1F* hr1 = c1->DrawFrame(-10,tblim,40,tulim);//the xy axis the fram (xmin,ymin,xmax,ymax)
	TH1F* hr1 = run(650,hr1);

	interval(88.);
	c2 = new TCanvas("c2","c2",200,10,700,500);
        TH1F* hr2 = c2->DrawFrame(-10,tblim,40,tulim);//the xy axis the fram (xmin,ymin,xmax,ymax)
	TH1F* hr2 = run(600,hr2);

	interval(88.);
        c3 = new TCanvas("c3","c3",200,10,700,500);
        TH1F* hr3 = c3->DrawFrame(-10,tblim,40,tulim);//the xy axis the fram (xmin,ymin,xmax,ymax)
	TH1F* hr3 = run(550,hr3);
	interval(88.);
        c4 = new TCanvas("c4","c4",200,10,700,500);
        TH1F* hr4 = c4->DrawFrame(-10,tblim,40,tulim);//the xy axis the fram (xmin,ymin,xmax,ymax)
	TH1F* hr4 = run(500,hr4);
	interval(88.);
        c5 = new TCanvas("c5","c5",200,10,700,500);
        TH1F* hr5 = c5->DrawFrame(-10,tblim,40,tulim);//the xy axis the fram (xmin,ymin,xmax,ymax)
	TH1F* hr5 = run(450,hr5);
	interval(88.);
        c6 = new TCanvas("c6","c6",200,10,700,500);	
        TH1F* hr6 = c6->DrawFrame(-10,tblim,40,tulim);//the xy axis the fram (xmin,ymin,xmax,ymax)
	TH1F* hr6 = run(400,hr6);
	interval(84.);
        c7 = new TCanvas("c7","c7",200,10,700,500);	
        TH1F* hr7 = c7->DrawFrame(-10,tblim,40,tulim);//the xy axis the fram (xmin,ymin,xmax,ymax)
	TH1F* hr7 = run(350,hr7);
	interval(68.);
        c8 = new TCanvas("c8","c8",200,10,700,500);	
        TH1F* hr8 = c8->DrawFrame(-10,tblim,40,tulim);//the xy axis the fram (xmin,ymin,xmax,ymax)
	TH1F* hr8 = run(300,hr8);
        c9 = new TCanvas("c9","c9",200,10,700,500);	
        TH1F* hr9 = c9->DrawFrame(-10,tblim,40,tulim);//the xy axis the fram (xmin,ymin,xmax,ymax)
	TH1F* hr9 = run(250,hr9);

//	close_tex(datafiledoc);
	cout << "completing plotting" << endl;
}
TH1F* run(Int_t wl, TH1F* hr){


//	create_table(wl,datafiledoc);

	TString a1 = "a1std";
	TString a2 = "a4std";
	TString a3 = "a5std";
	
	TString tit = Form("%i",wl);
	tit += " nm, transmittance ";
	hr->SetTitle(tit);
        hr->SetYTitle("Transmission % ");
        hr->SetXTitle(" ");

	//modify the input file name and
	//the setting or profile of the plotting
	//cout <<	"wl " << wl << endl;
	TGraphErrors* gr1 = plot(a1,wl,0);
        gr1->SetLineColor(1);
        //gr1->SetLineWidth(1);
        gr1->SetMarkerColor(1);
        gr1->SetMarkerStyle(1);
        //gr->SetTitle("a simple graph");
        //gr->GetXaxis()->SetTitle("Wavelength");
        //gr->GetYaxis()->SetTitle("Transmission");
        gr1->Draw("P");

	TGraphErrors* gr2 = plot(a2,wl,10);
	gr2->SetMarkerColor(2);
	gr2->SetMarkerStyle(1);
	gr2->SetLineColor(2);
	gr2->Draw("P");

	TGraphErrors* gr3 = plot(a3,wl,20);
	gr3->SetMarkerColor(3);
	gr3->SetMarkerStyle(1);
	gr3->SetLineColor(3);
	gr3->Draw("P");
	
	return hr;
	delete hr;
	cout << "completing running" << endl;
}



TGraphErrors* plot(TString file,Int_t wl, Int_t day){
	
//	gROOT->Reset();

	Int_t m(0);
	ifstream im,in;
	Double_t dx(0),dxx(0),dy(0),dyy(0),dex(0),dey(0);

	//decide the size of the matrix, don't modify it
	im.open(file);
	//cout << "start counting data size"<< endl;
	while(1){
		im >> dxx >> dyy;
		if(!im.good()) break;
		m++;
		//cout << m << " " << endl;
	}
	im.close();
	//cout << "the size is " << m << endl;
	const Int_t n = m;
	Double_t x[n],y[n],ex[n],ey[n];
	Double_t daya[1],xx[1],yy[1],exx[1],eyy[1];

	//Fill the data, don't modify it
	Int_t i = 0;
	in.open(file);
	while(1){
		in >> dx >> dy;
		if(!in.good()) break;
		x[i] = 200 + i;
		y[i] = dx;
		//error bar, up to yourself
		ex[i] = dex;
		ey[i] = dy;
		if(x[i] == wl){
			yy[0] = dx;
			exx[0] = 0;
			eyy[0] = dy;
			daya[0] = day;
		}
			
		//cout << x[i] << " " << y[i] << " " << ex[i] << " " << ey[i] << endl;
		i++;
	}
	in.close();
	//check the data array size
	if( n != i){
		cout << "sth. wrong with the data array size" << " i,n is " << i << "," << n << endl;
	}

//	cout << " trans is " << yy[0] << "   error bar is " << eyy[0] << endl;
	cout << yy[0] << " " << eyy[0] << endl;
	//plotting
	gr = new TGraphErrors(1,daya,yy,exx,eyy);
        //gr = new TGraph(n,x,y);

	return gr;
	delete gr;
	
}

void interval(Float_t tav){
	tulim = tav+5.;
	tblim = tav-5.;

	//cout << tav << " " << tulim << " " << tblim << endl;
}
	
