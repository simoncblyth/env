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

#define TOTALRAWDATANO 601
#define TOTALABSWL 270.0

double fitf(double *x, double *par) {

    double fitval = (par[0] - par[1])/(1 + exp((x[0] - par[2]) / par[3])) + par[1]
                    - (TOTALABSWL/x[0])*par[0];
    return fitval;

}

double fitf_exp(double* x, double* par) {

    double fitval = par[0]*exp(-(par[1]/x[0])) + par[2]*exp(-(par[3]/x[0]));
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

        if((attenuErrH > 0.1 && ((attenuErrH + attenu) < 10000.0))) {
            sizeCounter++;
            //cout << wl << " " << ior << " " << iorErr << " "
            //    << alpha << " " << alphaErr << " " << attenu << " "
            //    << attenuErrH << " " << attenuErrL << " " << tErr << " "
            //    << rErr << " " << solutionStatus << endl;
        }

    }
    cout << "sizeCounter" << sizeCounter << endl;

    fin.close();

    const int sizeArray = sizeCounter;
    double wlArray[sizeArray], iorArray[sizeArray];

    ifstream finData;
    finData.open("paras.dat");
    int dataCounter(0);
    for(int i=0;i<TOTALRAWDATANO;i++) {
        finData >> wl >> ior >> iorErr >> alpha >> alphaErr >> attenu
        >> attenuErrH >> attenuErrL >> tErr >> rErr >> solutionStatus;
        if((attenuErrH > 0.1 && ((attenuErrH + attenu) < 10000.0))) {
            wlArray[dataCounter] = wl;
            iorArray[dataCounter] = attenu;
            //cout << wl << " " << attenu << endl;
            dataCounter++;
        }
    }

    cout << "Reading in data......done!" << endl;

    TGraph* gr = new TGraph(sizeArray, wlArray, iorArray);
    return gr;

}

void attenuation_fit(void) {

    TCanvas *c1 = new TCanvas("c1","the fit canvas",500,400);

    TGraph* gr = readGraph();
    gr->Draw("a*");

    TF1 *func = new TF1("func",fitf,200.0,800.0,4);
    func->SetParameters(2.0, 3000.0, 290.0, 3.0);
    func->SetParNames("Lower_Attenu_Leng","Upper_Attenu_Leng","Cutting_Lambda","Delta");
    //TF1 *func = new TF1("func",fitf,200.0,370.0,4);
    //func->SetParameters(1.0, 10.0);
    //func->SetParNames("Amp","Thickness");
    gr->Fit(func,"r");

    //gr->Fit("pol2");

    func = gr->GetFunction("func");
    func->SetLineColor(kRed);
    func->SetLineWidth(1);
}

