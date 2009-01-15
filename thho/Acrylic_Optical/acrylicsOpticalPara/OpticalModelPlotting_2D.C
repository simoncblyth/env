// Ref:
// 1. Applied Optics / Vol.20.No.22 / 1 August 1990
//

#include <iostream>
#include <iomanip>
#include <complex>

#define PI 3.1415926
#define DELTA 1.0e-10
#define g_MAXLOOP 99
#define g_ACCURACY 1.0e-6

using namespace std;

void OpticalModelPlotting2D(Double_t nmin, Double_t nmax, 
                            Double_t kmin, Double_t kmax, Double_t d,
                            Double_t lambda,
                            const Double_t Tm, const Double_t Rm,
                            Int_t parNo) {

    

    TF2 *tt = new
        TF2("TRansmittance", GetTFormula,nmin,nmax,kmin,kmax,parNo);
    tt->SetParameters(d,lambda,Tm,Rm);

    TF2 *rr = new
        TF2("Reflectance", GetRFormula,nmin,nmax,kmin,kmax,parNo);
    rr->SetParameters(d,lambda,Tm,Rm);


    tt->Draw();
    rr->Draw();

}

Double_t GetTFormula(Double_t *x, Double_t *par) {

    return TFunc(x[0],x[1],par[0],par[1],par[2]);

}

Double_t GetRFormula(Double_t *x, Double_t *par) {

    return RFunc(x[0],x[1],par[0],par[1],
        GetOpticalModelTValue(GetIT(x[1],par[0],par[1]),GetFR(x[0],x[1])),
        par[3]);
}
/////////////////////////////////////////////////////////////////////////

// constrain of R
Double_t RFunc(Double_t n, Double_t k, Double_t d, Double_t lambda,
                Double_t Tmc, Double_t Rm) {

    Double_t y = GetOpticalModelRValue(Tmc, GetIT(k,d,lambda), GetFR(n,k))
                    - Rm;
    return y;

}

// constrain of T
Double_t TFunc(Double_t n, Double_t k, Double_t d,
                Double_t lambda, Double_t Tm) {

    Double_t y = GetOpticalModelTValue(GetIT(k,d,lambda),GetFR(n,k)) - Tm;
    return y;

}

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

