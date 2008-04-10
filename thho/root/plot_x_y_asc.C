// root macro for plotting x-y
// the code will read an asii file "test.asc" and then plotting
// the fromat of test.asc should be
//
// x y
// 1 2
// 3 4
// 3 3
// 
// 
//
//
//
//
#include "Riostream.h"
void main(void){
	gROOT->Reset();
	c1 = new TCanvas("c1","ccc",200,10,700,500);
	//c1->SetFillColor(42);
	//c1->SetGrid();
        TH1F *hr = c1->DrawFrame(0,0,10,10);//the xy axis the fram (xmin,ymin,xmax,ymax)
        hr->SetYTitle("Transmission % ");
        hr->SetXTitle("Wavelenth nm");
        c1->GetFrame()->SetFillColor(21);
        c1->GetFrame()->SetBorderSize(12);


	//modify the input file name and
	//the setting or profile of the plotting
	TGraphErrors* gr1 = plot("test.asc");
        gr1->SetLineColor(2);
        //gr->SetLineWidth(1);
        //gr->SetMarkerColor(4);
        //gr->SetMarkerStyle(21);
        //gr->SetTitle("a simple graph");
        //gr->GetXaxis()->SetTitle("Wavelength");
        //gr->GetYaxis()->SetTitle("Transmission");
        gr1->Draw("P");


	cout << "complete plotting" << endl;
}

TGraphErrors* plot(TString file){
	
	gROOT->Reset();

	Int_t m(0);
	ifstream im,in;
	Double_t dx(0),dxx(0),dy(0),dyy(0),dex(0),dey(0);

	//decide the size of the matrix, don't modify it
	im.open(file);
	cout << "start counting data size"<< endl;
	while(1){
		im >> dxx >> dyy;
		if(!im.good()) break;
		m++;
		cout << m << " " << endl;
	}
	im.close();
	cout << "the size is " << m << endl;
	const Int_t n = m;
	Double_t x[n],y[n],ex[n],ey[n];

	//Fill the data, don't modify it
	Int_t i = 0;
	in.open(file);
	while(1){
		in >> dx >> dy;
		if(!in.good()) break;
		x[i] = dx;
		y[i] = dy;
		//error bar, up to yourself
		ex[i] = dex;
		ey[i] = dey;
		i++;
	}
	//check the data array size
	if( n != i){
		cout << "sth. wrong with the data array size" << " i,n is " << i << "," << n << endl;
	}

	//plotting
	gr = new TGraphErrors(n,x,y,ex,ey);
        //gr = new TGraph(n,x,y);

	in.close();
	return gr;
	delete gr;
	
}


