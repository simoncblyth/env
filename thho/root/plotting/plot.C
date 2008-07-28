//input the x and y asci data file and then plot
//
//the only thing you should modify is the plot function
//
//plotg("file-contain-x-data","file-contain-y-data",weight value of y data)
//
//plotgac("xmm acrylic","ymm acrylic data",difference thick m,reflection fix %)
#include "Riostream.h"
void plot(void){
	gROOT->Reset();
	c1 = new TCanvas("c1","attenuation v.s. wavelength",200,10,700,500);
	c1->SetFillColor(42);
	c1->SetGrid();

	// draw a frame to define the range
	TH1F *hr = c1->DrawFrame(150,0,800,50);
	hr->SetYTitle("Attenuation m");
	hr->SetXTitle("Wavelength nm");
	c1->GetFrame()->SetFillColor(21);
	c1->GetFrame()->SetBorderSize(12);

	//modify these three lines
	TGraphErrors* gr1 = plotg("qe_wl","qe_100",70);
        gr1->SetLineColor(1);
	gr1->Draw("L");
	TGraphErrors* gr2 = plotg("gdls_wl","gdls_abs",1);
	gr2->SetLineColor(2);
	gr2->Draw("L");
	TGraphErrors* gr3 = plotg("gdls_wlfast","gdls_profast",10);
	gr3->SetLineColor(2);
	gr3->Draw("L");
	TGraphErrors *gr4 = plotg("gdls_wlre","gdls_prore",10);
	gr4->SetLineColor(2);
	gr4->Draw("L");
	TGraphErrors* gr5 = plotg("ls_wl","ls_abs",1);
	gr5->SetLineColor(5);
	gr5->Draw("L");
	TGraphErrors* gr6 = plotg("ls_wlfast","ls_profast",10);
	gr6->SetLineColor(5);
	gr6->Draw("L");
	TGraphErrors* gr7 = plotg("ls_wlre","ls_prore",10);
	gr7->SetLineColor(5);
	gr7->Draw("L");
	TGraphErrors* gr8 = plotg("mineral_wl","mineral_abs",1);
        gr8->SetLineColor(8);
	gr8->Draw("L");
	TGraphErrors* gr9 = plotgac("10mm","15mm",0.05,8);
        gr9->SetLineColor(9);
	gr9->Draw("L");
	TGraphErrors* gr10 = plotgt("10mm",5);
	gr10->SetLineColor(10);
	gr10->Draw("L");
	
}

TGraphErrors* plotg(TString filewl, TString fileabs, Int_t we){

	gROOT->Reset();

	//find out how many data and specify the size of array for plotting
	ifstream inxx;
	inxx.open(filewl);
	Double_t xx;
	Int_t sizex(0);
	while(1){
		inxx >> xx;
		if ( !inxx.good()) break;
		sizex++;
	}
	cout << "sizex = " << sizex << endl;
	ifstream inyy;
	inyy.open(fileabs);
	Double_t yy;
	Int_t sizey(0);
	while(1){
		inyy >> yy;
		if ( !inyy.good()) break;
		sizey++;
	}
	cout << "sizey = " << sizey << endl;
	if( sizex != sizey){
		cout << " input file " << filewl << " and " << fileabs << " data No. is incorrect!!!!!!!" << endl;
	}
	else{
		cout << " input file " << filewl << " and " << fileabs << " data No. is O.K." << endl;
	}



	//starting plotting
	ifstream inx;
	inx.open(filewl);
	ifstream iny;
	iny.open(fileabs);
	
	Int_t ix = 0;
	Double_t xxx,yyy;
       	const Int_t n = sizex;
	Double_t x[n];
	Double_t y[n];
	Double_t ex[n];
	Double_t ey[n];
	while (1){
		inx >> xxx ;
		if (!inx.good()) break;
		x[ix]=1200/xxx;
		ex[ix] = 0;
		ix++;
	}

	Int_t iy = 0;
	while (1){
		iny >> yyy;
		if(!iny.good()) break;
		y[iy] = yyy*we;
		ey[iy] = 0;
		iy++;
	}
	
	gr = new TGraphErrors(n,x,y,ex,ey);

	inx.close();
	iny.close();

	
	return gr;
	delete gr;

}

TGraphErrors* plotgt(TString file,Int_t we){

	gROOT->Reset();

	//find out how many data and specify the size of array for plotting
	ifstream inxx;
	inxx.open(file);
	Double_t xx,yy;
	Int_t sizex(0);
	while(1){
		inxx >> xx >> yy;
		if ( !inxx.good()) break;
		sizex++;
	}
	cout << "sizex = " << sizex << endl;

	inxx.close();
	
	//starting plotting
	ifstream inx;
	inx.open(file);
	
	Int_t ix = 0;
	Double_t xxx,yyy;
       	const Int_t n = sizex;
	Double_t x[n];
	Double_t y[n];
	Double_t ex[n];
	Double_t ey[n];
	while (1){
		inx >> xxx >> yyy;
		if (!inx.good()) break;
		x[ix]=xxx;
		y[ix]=yyy/we;
		ex[ix] = 0;
		ey[ix] = 0;
		ix++;
	}

	gr = new TGraphErrors(n,x,y,ex,ey);

	inx.close();

	
	return gr;
	delete gr;

}



TGraphErrors* plotgac(TString ac1, TString ac2, Double_t th, Double_t rf){

	gROOT->Reset();

	//find out how many data and specify the size of array for plotting
	ifstream in1;
	in1.open(ac1);
	Double_t x1;
	Double_t y1;
	Int_t size1(0);
	while(1){
		in1 >> x1 >> y1;
		if ( !in1.good()) break;
		size1++;
	}
	cout << "size1 = " << size1 << endl;
	ifstream in2;
	in2.open(ac2);
	Double_t x2;
	Double_t y2;
	Int_t size2(0);
	while(1){
		in2 >> x2 >> y2;
		if ( !in2.good()) break;
		size2++;
	}
	cout << "size2 = " << size2 << endl;
	if( size1 != size2){
		cout << " input file " << ac1 << " and " << ac2 << " data No. is incorrect!!!!!!!" << endl;
	}
	else{
		cout << " input file " << ac1 << " and " << ac2 << " data No. is O.K." << endl;
	}
	in1.close();
	in2.close();


	//starting plotting
	ifstream in11;
	in11.open(ac1);
	ifstream in22;
	in22.open(ac2);
	
	Int_t i1 = 0;
	Double_t x11e,y11e,x22e,y22e;
       	const Int_t n = size1;
	Double_t x11[n];
	Double_t y11[n];
	Double_t x22[n];
	Double_t y22[n];
	while (1){
		in11 >> x11e >> y11e;
		if (!in11.good()) break;
		x11[i1] = x11e;
		y11[i1] = y11e;
		i1++;
	}

	Int_t i2 = 0;
	while (1){
		in22 >> x22e >> y22e;
		if(!in22.good()) break;
		x22[i2] = x22e;
		y22[i2] = y22e;
		i2++;
	}
	
	Double_t ax[n];
	Double_t ay[n];
	Double_t aex[n];
	Double_t aey[n];
	for(Int_t ai=0;ai<n;ai++){
		if((y11[ai]==y22[ai]) || (y22[ai]==0)){
			if( x11[ai]==x22[ai]){
				ax[ai] = x11[ai];
			}
			else{
				cout << "sth. wrong in acrylic wavelength" << endl;
			}
		}
		else{
			ay[ai] = th/log((y11[ai]+rf)/(y22[ai]+rf));
			if( x11[ai]==x22[ai]){
				ax[ai] = x11[ai];
			}
			else{
				cout << "sth. wrong in acrylic wavelength" << endl;
			}
		}
		aex[ai]=0;
		aey[ai]=0;
	}
	gr = new TGraphErrors(n,ax,ay,aex,aey);

	in11.close();
	in22.close();

	
	return gr;
	delete gr;

}
