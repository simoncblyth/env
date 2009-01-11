// FresnelAnalysis.C
//
// A root code for evaluating the index of refaction n and
// the extinction coefficient k with the methoed like Ref 1.
//
// Usage:
//   Input the initial n, k, the thickness of sample d, the
// wavelength lambda, the measured reflectance R and the measured
// transmittance T and then the program will use Newton method
// to give you the optimized n and k which are consistent
// with the measured reflectance, transmittance within their
// experimental uncertainties.
//
// Ref:
// 1. Applied Optics / Vol.20.No.22 / 1 August 1990
//

#include <iostream>
#include <complex>

#define pi 3.1415926

using namespace std;

void FresnelAnalysis(Double_t n, Double_t k, Double_t d, Double_t lambda, Double_t Tm, Double_t Rm) {

    // initial Tmc and Rmc
    Double_t Tmc=0;
    Double_t Rmc=0;

    GetOpticalModelValue(Tmc, Rmc, GetIT(k,d,lambda), GetFR(n,k));


    printf("\nTm\tRm\tTmc\tRmc\t\n");
    printf("%f\t%f\t%f\t%f\n",Tm,Rm,Tmc,Rmc);

}


Double_t GetFR(Double_t n, Double_t k) {

    Double_t FR = ((n-1)*(n-1)+k*k)/((n+1)*(n+1)+k*k);
    return FR;

}

Double_t GetIT(Double_t k, Double_t d, Double_t lambda) {

    Double_t IT = exp((-4*pi*k*d)/lambda);
    return IT;

}

void GetOpticalModelRValue(Double_t &Tmc, Double_t &Rmc, Double_t IT, Double_t FR) {

    Rmc = FR*(1+IT*(Tmc));

}

void GetOpticalModelTValue(Double_t &Tmc, Double_t IT, Double_t FR) {

    Tmc = ((1-FR)*(1-FR)*IT)/(1-FR*FR*IT*IT);
    cout << Tmc << endl;

}

void GetOpticalModelValue(Double_t &Tmc, Double_t &Rmc, Double_t IT, Double_t FR) {

    GetOpticalModelTValue(Tmc, IT, FR);
    GetOpticalModelRValue(Tmc, Rmc, IT, FR);

}

void GetNandKwithNewtonMethod(Double_t n, Double_t k) {

}




