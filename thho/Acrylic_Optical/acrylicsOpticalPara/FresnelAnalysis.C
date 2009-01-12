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
#include <iomanip>
#include <complex>

#define PI 3.1415926
#define DELTA 1.0e-10
#define g_MAXLOOP 99
#define g_ACCURACY 1.0e-6

using namespace std;

void FresnelAnalysis(Double_t n, Double_t k, Double_t d,
            Double_t lambda, Double_t Tm, Double_t Rm) {

    // some printing out and value precision stuff
    cout.setf(ios::fixed | ios::showpoint);
    cout.precision(5);

    cout << "\nStarting using Newton method..........\n" << endl;
    cout<<"  loop       n       k       TFunc       RFunc         dx\
        dy" << endl;

    if(RunNewtonTwoD(n,k,g_MAXLOOP,g_ACCURACY,Tm,Rm,d,lambda)) {
        cout<< " \n successfully finding the root!" << endl;
        Double_t FinalTmc = GetOpticalModelTValue(GetIT(k,d,lambda),
                            GetFR(n,k));
        Double_t FinalRmc = GetOpticalModelRValue(FinalTmc,
                            GetIT(k,d,lambda),GetFR(n,k));
        cout<<"\n the model T is " << FinalTmc
            <<" and the model R is " << FinalRmc << endl;
    } else {
        cout << "\nfailed to find root " << endl;
    }

}

////////////////////////////////////////////////////////////////////////////
//////// Newton method /////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////

// 2D Newton method
Int_t RunNewtonTwoD(Double_t &n, Double_t &k, Int_t maxLoop,
                const Double_t accuracy, const Double_t Tm, 
                const Double_t Rm, const Double_t d,
                const Double_t lambda) {
    Double_t Tmf(0), Rmf(0), Tmn(0), Rmn(0), Tmk(0), Rmk(0);
    Double_t del(0), newn(0), newk(0), dn(0), dk(0);
    Int_t i(0);
    do
        {
        i++;
        InitializeFunc(n,k,d,Tm,Rm,Tmf,Rmf,Tmn,Rmn,Tmk,Rmk,lambda);
        GetCoupledJacobianFunc(n,k,Tmf,Rmf,Tmn,Rmn,Tmk,Rmk,
                                del,newn,newk,dn,dk);
        cout<<setw(3)<<i<<setw(12)<<n<<setw(12)<<k<<setw(12)<<setw(12)
            <<Tmf<<setw(12)<<Rmf<<setw(12)<<dn<<setw(12)<<dk<<endl;
        cout << "accuracy for n is " << fabs(dn) << " for k is "
                << fabs(dk) << "\n" << endl;
        }
    while(fabs(dn) >= accuracy && fabs(dk) >= accuracy && (--maxLoop));

    // return the maxLoop as the flag to express finding
    // a root successfully
    return maxLoop;

}

// constrain of R
Double_t RFunc(Double_t n, Double_t k, Double_t d, Double_t lambda,
                Double_t Tmc, Double_t Rm) {

    Double_t y = GetOpticalModelRValue(Tmc, GetIT(k,d,lambda), GetFR(n,k))
                    - Rm;
    return y;

}

Double_t RPartialn(Double_t n, Double_t k, Double_t d, Double_t lambda,
                Double_t Tmc, Double_t Rm) {

    Double_t y = (RFunc(n+DELTA/2,k,d,lambda,Tmc,Rm)
                -RFunc(n-DELTA/2,k,d,lambda,Tmc,Rm))/DELTA;
    return y;

}

Double_t RPartialk(Double_t n, Double_t k, Double_t d, Double_t lambda,
                Double_t Tmc, Double_t Rm) {

    Double_t y = (RFunc(n+DELTA/2,k,d,lambda,Tmc,Rm)
                -RFunc(n-DELTA/2,k,d,lambda,Tmc,Rm))/DELTA;

    return y;
}
// constrain of T
Double_t TFunc(Double_t n, Double_t k, Double_t d,
                Double_t lambda, Double_t Tm) {

    Double_t y = GetOpticalModelTValue(GetIT(k,d,lambda),GetFR(n,k)) - Tm;
    return y;

}

Double_t TPartialn(Double_t n, Double_t k, const Double_t d,
                    const Double_t lambda, const Double_t Tm) {

    Double_t y = (TFunc(n+DELTA/2,k,d,lambda,Tm)
                - TFunc(n-DELTA/2,k,d,lambda,Tm)) / DELTA;
    return y;

}

Double_t TPartialk(Double_t n, Double_t k, const Double_t d,
                    const Double_t lambda, const Double_t Tm) {

    Double_t y = (TFunc(n,k+DELTA/2,d,lambda,Tm)
                - TFunc(n,k-DELTA/2,d,lambda,Tm)) / DELTA;
    return y;

}

void InitializeFunc(Double_t &n, Double_t &k,const Double_t d,
                const Double_t Tm, const Double_t Rm,
                Double_t &Tmf, Double_t &Rmf, Double_t &Tmn, Double_t &Rmn,
                Double_t &Tmk, Double_t &Rmk, const Double_t lambda) {

        Tmf = TFunc(n,k,d,lambda,Tm);
        Rmf = RFunc(n,k,d,lambda,
                GetOpticalModelTValue(GetIT(k,d,lambda),GetFR(n,k)),Rm);
        Tmn = TPartialn(n,k,d,lambda,Tm);
        Tmk = TPartialk(n,k,d,lambda,Tm);
        Rmn = RPartialn(n,k,d,lambda,
                GetOpticalModelTValue(GetIT(k,d,lambda),GetFR(n,k)),Rm);
        Rmk = RPartialk(n,k,d,lambda,
                GetOpticalModelTValue(GetIT(k,d,lambda),GetFR(n,k)),Rm);

}

void GetCoupledJacobianFunc(Double_t &n,Double_t &k,
            Double_t &Tmf, Double_t &Rmf, Double_t &Tmn, Double_t &Rmn,
            Double_t &Tmk, Double_t &Rmk, Double_t &del, Double_t &newn,
            Double_t &newk, Double_t &dn, Double_t &dk) {

    del = Tmn*Rmk - Tmk*Rmn;
    dn = (Tmk*Rmf - Tmn*Rmf)/del;
    dk = (Tmf*Rmn - Tmn*Rmf)/del;
    newn = n + dn;
    newk = k + dk;

    n = newn;
    k = newk;
    
}

////////////////////////////////////////////////////////////////////////////
//////// Fresnel relationship //////////////////////////////////////////////
Double_t GetFR(Double_t n, Double_t k) {

    // simple formula to check the code
    Double_t FR = n+2;

    //Double_t FR = ((n-1)*(n-1)+k*k)/((n+1)*(n+1)+k*k);
    return FR;

}

Double_t GetIT(Double_t k, Double_t d, Double_t lambda) {

    // simple formula to check the code
    Double_t IT = k+5;

    //Double_t IT = exp((-4*PI*k*d)/lambda);
    return IT;

}
////////////////////////////////////////////////////////////////////////////
//////// Optical Model /////////////////////////////////////////////////////
Double_t GetOpticalModelRValue(Double_t Tmc, Double_t IT, Double_t FR) {

    // simple relation to check the code
    Double_t y = FR;

    //Double_t y = FR*(1+IT*Tmc);
    return y;

}

Double_t GetOpticalModelTValue(Double_t IT, Double_t FR) {

    // simple relation to check the code
    Double_t y = IT;

    //Double_t y = ((1-FR)*(1-FR)*IT)/(1-FR*FR*IT*IT);
    return y;

}

