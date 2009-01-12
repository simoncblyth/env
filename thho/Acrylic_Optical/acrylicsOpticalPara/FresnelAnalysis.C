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
#define THICKNESS 10
#define g_MAXLOOP 7
#define g_ACCURACY 1.0e-6

using namespace std;

void FresnelAnalysis(Double_t n, Double_t k, Double_t d,
            Double_t lambda, Double_t Tm, Double_t Rm) {

    // some printing out and value precision stuff
    cout.setf(ios::fixed | ios::showpoint);
    cout.precision(5);

    // initial Tmc and Rmc
    Double_t Tmc=0;
    Double_t Rmc=0;

    GetOpticalModelValue(Tmc, Rmc, GetIT(k,d,lambda), GetFR(n,k));
    printf("\nTm\tRm\tTmc\tRmc\t\n");
    printf("%f\t%f\t%f\t%f\n",Tm,Rm,Tmc,Rmc);

    cout << "\nStarting using Newton method..........\n" << endl;

//    if(RunNewtonOneD(n, k, 7, 1.0e-6,Rm))
//       cout << "\nroot n = " << n << ", test of f(x) = " << RFunc(n,k,Rm);
//    else cout << "\nfailed to find root ";

    cout<<"  loop       n       k       TFunc       RFunc         dx\          dy" << endl;

    if(RunNewtonTwoD(n,k,g_MAXLOOP,g_ACCURACY,Tm,Rm,d,lambda)) {
        cout<< " \n successfully finding the root!" << endl;
        ////////////////////////////////////////////
        /////////// wrong!!!!!!!!!!!!!!!!!!!!!!!!!
        ////////////////////////////////////////////
        Double_t FinalTmc = GetIT(k,d,lambda);
        Double_t FinalRmc = GetFR(n,k);
        cout<<"\n the model T is " << FinalTmc
            <<" and the model R is " << FinalRmc << endl;
    } else {
        cout << "\nfailed to find root " << endl;
    }

}

////////////////////////////////////////////////////////////////////////////
//////// Newton method /////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////

// 1D Newton method to check whether the result is resonable or not
Int_t RunNewtonOneD(Double_t &x, const Double_t k, Int_t max_loop,
                const Double_t accuracy, const Double_t Rm) {
    Double_t term;
    do
        {
         // calculate next term f(x) / f'(x) then subtract from current root
         term = RFunc(x, k, Rm) / RFdiv(x, k, Rm);
         x = x - term;// new root
         cout << "new n is " << x << "\taccu is " << fabs(term/x) << endl;
         
        }
    // check if term is within required accuracy or loop limit is exceeded
    while ((fabs(term / x) > accuracy) && (--max_loop));
    cout << "final max_loop " << max_loop << endl;
    return max_loop;
}

// 2D Newton method
Int_t RunNewtonTwoD(Double_t &n, Double_t &k, Int_t maxLoop,
                const Double_t accuracy, const Double_t Tm, 
                const Double_t Rm, const Double_t d,
                const Double_t lambda) {
    Double_t term;
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
        cout << "accuracy for n is " << fabs(dn/n) << " for k is "
                << fabs(dk/k) << "\n" << endl;
        }
    while(fabs(dn/n) >= accuracy && fabs(dk/k) >= accuracy && (--maxLoop));

    // return the maxLoop as the flag to express finding
    // a root successfully
    return maxLoop;

}

// constrain of R
Double_t RFunc(Double_t n, Double_t k, Double_t Rm) {

    // wrong!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    return GetFR(n,k) - Rm;

}

Double_t RFdiv(Double_t n, Double_t k, Double_t Tm) {

    return RFpartialn(n,k,Tm);

}

Double_t RPartialn(Double_t n, Double_t k, Double_t Tm) {

    Double_t y = (RFunc(n+DELTA/2,k,Tm)-RFunc(n-DELTA/2,k,Tm))/DELTA;
    return y;

}

Double_t RPartialk(Double_t n, Double_t k, Double_t Tm) {

    Double_t y = (RFunc(n,k+DELTA/2,Tm)-RFunc(n,k-DELTA/2,Tm))/DELTA;
    return y;
}

// wrong !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1
Double_t TFunc(Double_t n, Double_t k, Double_t d, Double_t lambda, Double_t Tm) {

    return GetIT(k, d, lambda) -Tm;

}

Double_t TPartialn(Double_t k,const Double_t d, const Double_t lambda) {

    Double_t y = 0;
    return y;

}

Double_t TPartialk(Double_t n, Double_t k,const Double_t d,
                    const Double_t lambda, const Double_t Tm) {

    Double_t y = (TFunc(n,k+DELTA/2,d,lambda,Tm)
                - TFunc(n,k-DELTA/2,d,lambda,Tm)) / DELTA;
    return y;

}

void InitializeFunc(Double_t &n, Double_t &k,const Double_t d,
                const Double_t Tm, const Double_t Rm,
                Double_t &Tmf, Double_t &Rmf, Double_t &Tmn, Double_t &Rmn,
                Double_t &Tmk, Double_t &Rmk, const Double_t lambda) {

        // wrong!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        Tmf = TFunc(n,k,d,lambda,Tm);
        Rmf = RFunc(n,k,Rm);
        Tmn = TPartialn(k,THICKNESS,lambda);
        Rmn = RPartialn(n,k,Tm);
        Tmk = TPartialk(n,k,THICKNESS,lambda,Tm);
        Rmk = RPartialk(n,k,Rm);

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

    Double_t FR = ((n-1)*(n-1)+k*k)/((n+1)*(n+1)+k*k);
    return FR;

}

Double_t GetIT(Double_t k, Double_t d, Double_t lambda) {

    Double_t IT = exp((-4*PI*k*d)/lambda);
    return IT;

}
////////////////////////////////////////////////////////////////////////////
//////// Optical Model /////////////////////////////////////////////////////
void GetOpticalModelRValue(Double_t &Tmc, Double_t &Rmc,
                Double_t IT, Double_t FR) {

    Rmc = FR*(1+IT*(Tmc));

}

void GetOpticalModelTValue(Double_t &Tmc, Double_t IT, Double_t FR) {

    Tmc = ((1-FR)*(1-FR)*IT)/(1-FR*FR*IT*IT);

}

void GetOpticalModelValue(Double_t &Tmc, Double_t &Rmc, Double_t IT, Double_t FR) {

    GetOpticalModelTValue(Tmc, IT, FR);
    GetOpticalModelRValue(Tmc, Rmc, IT, FR);

}
////////////////////////////////////////////////////////////////////////////




