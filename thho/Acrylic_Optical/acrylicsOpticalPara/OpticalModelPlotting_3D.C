// Quick drawing to see the pattern of an acrylic optical model
// given n ,k, and intensity ratio range
//
// Ref:
// 1. Applied Optics / Vol.20.No.22 / 1 August 1990
//

#define PI 3.1415926

using namespace std;

void OpticalModelPlotting(Double_t nmin, Double_t nmax, 
                            Double_t kmin, Double_t kmax, Double_t d,
                            Double_t lambda,
                            Double_t rtmin, Double_t rtmax, Int_t par) {

    TF3 *tt = new
        TF3("TRansmittance", GetTFormula,
                nmin,nmax,kmin,kmax,rtmin,rtmax,par);
    tt->SetParameters(d,lambda);

    TF3 *rr = new
        TF3("Reflectance", GetRFormula,
                nmin,nmax,kmin,kmax,rtmin,rtmax,par);
    rr->SetParameters(d,lambda);


    tt->Draw();
    //rr->Draw();

}

Double_t GetTFormula(Double_t *x, Double_t *par) {

    return GetOpticalModelTValue(GetIT(x[1],par[0],par[1]),
                GetFR(x[0],x[1]))
            -x[2];

}

Double_t GetRFormula(Double_t *x, Double_t *par) {

    return GetOpticalModelRValue(
            GetOpticalModelTValue(GetIT(x[1],par[0],par[1]),
                                    GetFR(x[0],x[1])),
                GetIT(x[1],par[0],par[1]),
                GetFR(x[0],x[1]))
            - x[2];

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

