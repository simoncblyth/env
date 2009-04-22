//
// 1.
// Quick drawing to see the pattern of an acrylic optical model
// given n and k.
//
// usage:
//  shell prompt> root
//  root cint> .L OpticalModelPlotting_2D
//  root cint> OpticalModelPlotting2D(1.1,2.0, 0.0,2.0, 10,400,2)
//
// 2.
// Quick comfirm measured T and R with known n and k
//
// usage:
//  shell prompt> root
//  root cint> .L OpticalModelPlotting_2D
//  root cint> .L GetMeasureValue(alpha-cm^-1,n,d-mm,lambda-nm)
//
//
// Ref:
// 1. Applied Optics / Vol.20.No.22 / 1 August 1990
//
#define PI 3.1415926

using namespace std;

void OpticalModelPlotting2D(Double_t nmin, Double_t nmax, 
                            Double_t kmin, Double_t kmax, Double_t d,
                            Double_t lambda,
                            Int_t parNo, Double_t nfix, Double_t kfix) {

    TCanvas *c1 = new TCanvas(
        "c1","Optical Model T and R",200,10,700,900);
    c1->Divide(2,3);
    //title = new TPaveText(.2,0.96,.8,.995);
    //title->AddText("Optical Model T and R");
    //title->Draw();

    //pad1 = new TPad("pad1","TransmittancePad",0.03,0.50,0.98,0.95,21);
    //pad2 = new TPad("pad2","ReflectancePad",0.03,0.02,0.98,0.48,21);
    //pad1->Draw();
    //pad2->Draw();

    c1->cd(1);
    TF2 *tt = new
        TF2("Transmittance", GetTFormula,nmin,nmax,kmin,kmax,parNo);
    //pad1->cd();
    //pad1->SetPhi(-80);
    tt->SetParameters(d,lambda);
    // color mesh
    tt->Draw("surf1");
    // gouraud shading
    //tt->Draw("surf4");

    c1->cd(2);
    TF2 *rr = new
        TF2("Reflectance", GetRFormula,nmin,nmax,kmin,kmax,parNo);
    //pad2->cd();
    rr->SetParameters(d,lambda);
    rr->Draw("surf1");

    c1->cd(3);
    TF1 *ttk = new
        TF1("Transmittance of fix n",GetTFormulaK,kmin,kmax,parNo+1);
    ttk->SetParameters(d,lambda,nfix);
    ttk->Draw();

    c1->cd(4);
    TF1 *rrk = new
        TF1("Reflectance of fix n",GetRFormulaK,kmin,kmax,parNo+1);
    rrk->SetParameters(d,lambda,nfix);
    rrk->Draw();

    c1->cd(5);
    TF1 *ttn = new
        TF1("Transmittance of fix k",GetTFormulaN,nmin,nmax,parNo+1);
    ttn->SetParameters(d,lambda,kfix);
    ttn->Draw();

    c1->cd(6);
    TF1 *rrn = new
        TF1("Reflectance of fix k",GetRFormulaN,nmin,nmax,parNo+1);
    rrn->SetParameters(d,lambda,kfix);
    rrn->Draw();



}

// simple caculate with given n and alpha
void GetMeasureValue(Double_t alpha, Double_t n,
                    Double_t d, Double_t lambda) {

    lambda = lambda*1.0e-6; // nm --> mm
    alpha = alpha*0.1; //cm-1 --> mm
    Double_t k = (alpha*lambda)/(4.*PI);
    Double_t Tmc = GetOpticalModelTValue(GetIT(k,d,lambda),GetFR(n,k));
    Double_t Rmc = GetOpticalModelRValue(Tmc, GetIT(k,d,lambda),GetFR(n,k));

    cout << endl;
    cout << "k\t\t" << "n\t" << "d\t" << "lambda\t" << endl;
    cout << k << "\t" << n << "\t" << d << "\t" << lambda << "\t" << endl;
    cout << "---------------------" << endl;
    cout << "IT\t" << GetIT(k,d,lambda) << endl;
    cout << "FR\t" << GetFR(n,k) << endl;
    cout << "Tmc\t" << Tmc << endl;
    cout << "Rmc\t" << Rmc << "\tTmc+Rmc\t" << Tmc+Rmc << endl;

}

Double_t GetTFormula(Double_t *x, Double_t *par) {

    return GetOpticalModelTValue(
                GetIT(x[1],par[0],par[1]),GetFR(x[0],x[1]));

}

Double_t GetRFormula(Double_t *x, Double_t *par) {

    return GetOpticalModelRValue(
        GetOpticalModelTValue(GetIT(x[1],par[0],par[1]),GetFR(x[0],x[1])),
        GetIT(x[1],par[0],par[1]),
        GetFR(x[0],x[1]));

}
Double_t GetTFormulaN(Double_t *x, Double_t *par) {

    return GetOpticalModelTValue(
                GetIT(par[2],par[0],par[1]),GetFR(x[0],par[2]));

}

Double_t GetRFormulaN(Double_t *x, Double_t *par) {

    return GetOpticalModelRValue(
        GetOpticalModelTValue(GetIT(par[2],par[0],par[1]),GetFR(x[0],par[2])),
        GetIT(par[2],par[0],par[1]),
        GetFR(x[0],par[2]));

}
Double_t GetTFormulaK(Double_t *x, Double_t *par) {

    return GetOpticalModelTValue(
                GetIT(x[0],par[0],par[1]),GetFR(par[2],x[0]));

}

Double_t GetRFormulaK(Double_t *x, Double_t *par) {

    return GetOpticalModelRValue(
        GetOpticalModelTValue(GetIT(x[0],par[0],par[1]),GetFR(par[2],x[0])),
        GetIT(x[0],par[0],par[1]),
        GetFR(par[2],x[0]));

}
/////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////
//////// Fresnel relationship //////////////////////////////////////////////
Double_t GetFR(Double_t n, Double_t k) {

    // simple formula to check the code
    //Double_t FR = n+2;

    Double_t FR = ((n-1)*(n-1)+k*k)/((n+1)*(n+1)+k*k);
    return FR;

}

Double_t GetIT(Double_t k, Double_t d, Double_t lambda) {

    // simple formula to check the code
    //Double_t IT = k+5;

    Double_t IT = exp((-4*PI*k*d)/lambda);
    return IT;

}
////////////////////////////////////////////////////////////////////////////
//////// Optical Model /////////////////////////////////////////////////////
Double_t GetOpticalModelRValue(Double_t Tmc, Double_t IT, Double_t FR) {

    // simple relation to check the code
    //Double_t y = FR;

    // Model 1
    Double_t y = FR*(1+IT*Tmc);

    // Model 2
    //Double_t y = FR*(1+IT*Tmc);
    return y;

}

Double_t GetOpticalModelTValue(Double_t IT, Double_t FR) {

    // simple relation to check the code
    //Double_t y = IT;

    // Model 1
    Double_t y = ((1-FR)*(1-FR)*IT)/(1-FR*FR*IT*IT);

    // Model 2
    //Double_t y = (1 -FR)*IT;
    return y;

}
