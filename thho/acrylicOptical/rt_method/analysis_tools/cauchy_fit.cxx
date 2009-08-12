/*******************************************************\

cauchy_fit.cxx

A root script to model the measurement
refractive index values by Cauchy equation.

There is also Sellmeier equation but doesn't be used
because the modeling is bad for my measurment.

Usage: root cauchy_fit.cxx
input measurement file name: para.dat
input data format:
     wavelength(nm) refractive_index

Author: Taihsiang
Date: June, 15, 2009

\*******************************************************/

//#define ATTCONSTRAIN 10000.0 // unit: mm, the reasonable attenuation up limit

#define TOTALRAWDATANO 601

double fitf(double *x, double *par) {

    double fitval = par[0] + par[1]/(x[0]*x[0]);
    return fitval;

}

double fitfSell(double *x, double *par) {

    double fitval = sqrt(1.0 + par[0]*x[0]*x[0]/(x[0]*x[0] - par[1])
                            + par[2]*x[0]*x[0]/(x[0]*x[0] - par[3])
                            + par[4]*x[0]*x[0]/(x[0]*x[0] - par[5]));
    return fitval;
}

TGraph* readGraph(void) {

    ifstream fin;
    fin.open("paras.dat");

    int sizeCounter(0);
    double wl(0), ior(0);
    double iorErr(0), alpha(0), alphaErr(0);
    double attenu(0), attenuErrH(0), attenuErrL(0);
    double tErr(0), rErr(0), solutionStatus;

    while(1) {
        fin >> wl >> ior >> iorErr >> alpha >> alphaErr >> attenu
        >> attenuErrH >> attenuErrL >> tErr >> rErr >> solutionStatus;
        if(!fin.good()) break;

        #ifdef ATTCONSTRAIN
        if((attenuErrH > 0.1 && ((attenuErrH + attenu) < ATTCONSTRAIN))) sizeCounter++;
        #endif

        #ifndef ATTCONSTRAIN
        sizeCounter++;
        #endif
    }
    cout << "sizeCounter " << sizeCounter << endl;


    fin.close();

    const int sizeArray = sizeCounter;
    double wlArray[sizeArray], iorArray[sizeArray];

    ifstream finData;
    finData.open("paras.dat");
    int dataCounter(0);
    for(int i=0;i<TOTALRAWDATANO;i++) {
        finData >> wl >> ior >> iorErr >> alpha >> alphaErr >> attenu
        >> attenuErrH >> attenuErrL >> tErr >> rErr >> solutionStatus;
        #ifdef ATTCONSTRAIN
        if((attenuErrH > 0.1 && ((attenuErrH + attenu) < ATTCONSTRAIN))) {
            wlArray[dataCounter] = wl;
            iorArray[dataCounter] = ior;
            //cout << wl << " " << ior << endl;
            dataCounter++;
        }
        #endif

        #ifndef ATTCONSTRAIN
        //cout << "ATTCONSTRAIN is not defined" << endl;
        wlArray[i] = wl;
        iorArray[i] = ior;
        //cout << wl << " " << ior << endl;
        #endif
    }
    cout << "dataCounter " << dataCounter << endl;

    TGraph* gr = new TGraph(sizeArray, wlArray, iorArray);
    return gr;

}

void cauchy_fit(void) {

    TCanvas *c1 = new TCanvas("c1","the fit canvas",500,400);
    gStyle->SetOptFit(1111);

    TGraph* gr = readGraph();
    gr->SetTitle("Cauchy Equation and Measured n");
    gr->Draw("a*");

    TF1 *func = new TF1("func",fitf,200.0,800.0,2);
    //TF1 *func = new TF1("func",fitfSell,300.0,700.0,6);
    //TF1 *func = new TF1("func", "1.5 ++ 0.005*(x*x)", 200, 800);

    func->SetParameters(1.5,0.005);
    func->SetParNames("Constant_A","Constant_B");
    //func->SetParameters(1.0, 1.03, 0.23, 1.01, 0.006, 0.02, 103.6);
    //func->SetParNames("B1","C1","B2","C2","B3","C3");
    gr->Fit(func,"r");
    //gr->Fit("pol3");

    func = gr->GetFunction("func");
    func->SetLineColor(kRed);
    func->SetLineWidth(1);

}

