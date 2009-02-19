// FresnelAnalysis.C
//
// A root code for evaluating the index of refaction n and
// the extinction coefficient k with the methoed like Ref 1.
//
// Then the program will use Newton method
// to give you the optimized n and k which are consistent
// with the measured reflectance, transmittance within their
// experimental uncertainties. The algorithm to optimize
// for n is RT method and for k is RTT' method.
//
// RT finds out the optimized n and k with 2D Newton method.
// Set previous optimized n as initial n parameter for RTT'
// method and then RTT' finds out k with 1D Newton method.
//
// Usage:
//   Input
//
// the initial n,
// absorption coefficient alpha,
// the thickness of sample d,
// the wavelength lambda,
// the measured reflectance R
// and the measured transmittance T.
//
//
// shell prompt> root
// root cint> .L FresnelAnalysis.C
// root cint> SucApp(1.505,0.009,405.0,
//                      10.14,0.92253096,0.07979050,
//                      14.80,0.92029215,0.08033188)
//
// SucApp(index of refraction, alpha--cm-1, wavelength--nm,
//          thickness of thinner one -- mm,
//          thinner one transmittance, thinner one reflectance
//          thickness of thicker one -- mm,
//          thicker one transmittance, thicker one reflectance)
//
// Ref:
// 1. Applied Optics / Vol.20.No.22 / 1 August 1990
//
//
// Author: Taihsiang Ho
// Contact: thho@hep1.phys.ntu.edu.tw
// Date: 2009.Jan
//
// TODO: 1. A function processes successive approach with different
//          searching initial value of n or k
//          when the Newton method find a phsical meaningless n or k
//       2. Check the consistent n and k for thin and thick sample.
//       3. Stopping when finding process failed not compeletely defined.
//
//


#include <iostream>
#include <iomanip>
#include <complex>

#define PI 3.1415926
#define DELTA 1.0e-11
#define g_MAXLOOP 99
#define g_ACCURACY 1.0e-6

using namespace std;

// n and alpha value container
typedef struct nacon
{
    Double_t n;
    Double_t alpha;

};



// read in data and analyzing it
//
// data format
//
// wl(nm) TorT
// 800.0  0.920
// 790.0  0.915
//

void AnalyzData(TString tfile, TString rfile,
                TString thtfile, TString thrfile,
                Double_t n, Double_t alpha,
                Double_t thin,Double_t thick) {

    Double_t wldataContainer[601]={0.0};
    Double_t tdataContainer[601]={0.0};
    Double_t rdataContainer[601]={0.0};
    Double_t thtdataContainer[601]={0.0};
    Double_t thrdataContainer[601]={0.0};
    ReadData(tfile, wldataContainer, tdataContainer);
    ReadData(rfile, wldataContainer, rdataContainer);
    ReadData(thtfile, wldataContainer, thtdataContainer);
    ReadData(thrfile, wldataContainer, thrdataContainer);

    //Fakedata(wldataContainer, tdataContainer, rdataContainer, thtdataContainer, thrdataContainer);
    //if(CheckDataFormate(tdataContainer,rdataContainer,thtdataContainer,thrdataContainer)==0) {
    cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << endl;
    if(2>1){
        for(Int_t i=0;i<601;i++){
            if(SucApp(n, alpha, wldataContainer[i],
                    thin, tdataContainer[i], rdataContainer[i],
                    thick, thtdataContainer[i], thrdataContainer[i])==0) {
                cout << "YES!!" << endl;
            } else break;
        }
    } else break;

}

void Fakedata(Double_t a[], Double_t b[],Double_t c[],Double_t d[],Double_t e[]) {
    for(Int_t i=0;i<601;i++){
            a[i]=10;
            b[i]=10;
            b[i]=10;
            d[i]=10;
            e[i]=10;
    }
}

/*
Int_t CheckDataFormat(Double_t tdataContainerSize[], Double_t rdataContainer[],
                        Double_t thtdataContainerSize[], Double_t thrdataContainer[]) {

    if((tdataContainer == rdataContainer)
        && (rdataContainer == thtdataContainer)
        && (thtdataContainer == thrdataContainer)) {
        cout << "The numbers between files are consistent." << endl;
    } else {
        cout << "The numbers between files are different." << endl;
        break;
    }

    return 0;

}
*/
// return the size of array to contain data
Int_t CheckDataSize(TString file) {

    ifstream inputSizeFile;
    inputSizeFile.open(file);

    Int_t inputSizeCount(0);
    while(1) {
        Double_t i,j;
        inputSizeFile >> i >> j;
            if(!inputSizeFile.good()) break;
            inputSizeCount++;
        }
    inputSizeFile.close();
    return inputSizeCount;

}
void ReadData(TString file, Double_t wl[], Double_t data[]){

    ifstream inputDataFile;

    //Int_t tmpsize = CheckDataSize(file);
    const Int_t inputSize = 601;

    //cout << "size is " << inputSizeCount << endl;
    //cout << "size is " << inputSize << endl;

    Double_t wlContain[inputSize];
    Double_t trContain[inputSize];

    Int_t fillingCounter(0);
    inputDataFile.open(file);
        while(1) {
            Double_t i,j;
            inputDataFile >> i >> j;
            if(!inputDataFile.good()) break;
            wlContain[fillingCounter] = i;
            trContain[fillingCounter] = j*0.01; // xx.xx% -> 0.xxxx
            fillingCounter++;
        }
    inputDataFile.close();

    CopyArray(wlContain,wl,inputSize);
    CopyArray(trContain,data,inputSize);


    cout << "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX" << endl;

}

// copy a array A to another array b
void CopyArray(Double_t a[], Double_t b[], Int_t size) {
    for(Int_t i=0;i<size;i++) {
        b[i] = a[i];
    }
} 

// successive approach
Int_t SucApp(Double_t n, Double_t alpha, Double_t lambda,
            Double_t thin, Double_t thinTm, Double_t thinRm,
            Double_t thick, Double_t thickTm, Double_t thickRm ) {

    nacon nac;
    Int_t maxLoop = g_MAXLOOP;
    Double_t deltan(0);
    Double_t deltaalpha(0);
    Double_t thinn(0);
    Double_t thickn(0)
    Double_t deltaalpha(0);
    Double_t thinalpha(0);
    Double_t thickalpha(0);

    // unit staff -- units should unite.
    lambda = lambda*1.0e-6; // unit: nm->mm
    alpha = alpha*0.1; // unit: cm-1 -> mm-1

    do
        {
        FresnelAnalysisnk(n,alpha, thin, lambda, thinTm, thinRm, &nac);
        thinn = nac.n;
        thinalpha = nac.alpha;
        //cout << "1 n " << nac.n << " k " << (lambda*nac.alpha)/(4*PI)
        //<< " alpha  " << nac.alpha/0.1<<"cm-1" <<endl;
        FresnelAnalysisk(thinn,nac.alpha, thick,lambda,thickTm, thickRm, &nac);
        thickn = nac.n;
        thickalpha = nac.alpha;
        //cout << "1 n " << nac.n << " k " << (lambda*nac.alpha)/(4*PI)
        //<<" alpha  " << nac.alpha/0.1<<"cm-1" << endl;

        deltan = thinn - thickn;
        deltaalpha = thinalpha - thickalpha;

        cout << "\n*****************************************************" << endl;
        cout << "*****************************************************" << endl;
        cout << "*****************************************************" << endl;
        cout << "\nFINAL RESULT" << endl;
        cout << "   n         alpha            delta-n        delta-alpha"
            << endl;
        cout << setw(8) << nac.n << setw(16)
            << nac.alpha << setw(16)
            << deltan << setw(16)
            << deltaalpha << endl << endl;
        PrintTandR(nac.n,nac.alpha,thin,lambda);
        PrintTandR(nac.n,nac.alpha,thick,lambda);

        }
    while((nac.n<1 || nac.n==1 || fabs(deltan) > 0.01 || nac.alpha<0 || nac.alpha==0 || fabs(deltaalpha) > 0.001) && (--maxLoop));

    if(maxLoop){        
        cout << "\n\n\nSUCCESSESSFULLY FINDING A CONSISTENT N AND K!!! :)\n\n\n" << endl;
    } else {
        cout << "\n\n\nFAILED TO FIND A COSISTENT N AND K!!! :(\n\n\n" << endl;
    }
    cout << "*****************************************************" << endl;
    cout << "*****************************************************" << endl;
    cout << "*****************************************************\n\n\n" << endl;

    return 0;
}

void FresnelAnalysisk(Double_t n, Double_t alpha, Double_t d, 
            Double_t lambda, Double_t Tm, Double_t Rm, nacon* nac) { 
 
    // some printing out and value precision stuff 
    //cout.setf(ios::fixed | ios::showpoint); 
    //cout.precision(5); 

    Double_t k = (lambda*alpha)/(4*PI); 

    cout << endl << "---------------------------------------" << endl;
    cout << "\nStarting using 1D Newton method for k only...\n" << endl; 
    cout<<"  loop       n       k       TFunc       RFunc         dx     dy" << endl; 
 
    if(RunNewtonOneD(n,k,g_MAXLOOP,g_ACCURACY,Tm,Rm,d,lambda)) { 
        cout<< " \n successfully finding the root!\n" << endl;
        alpha = (k*4.0*PI)/(lambda);
        PrintTandR(n,alpha,d,lambda);
    } else { 
        cout << "\nfailed to find root " << endl; 
    } 
    cout << endl << "---------------------------------------" << endl;

    nac->n = n;
    nac->alpha = alpha;
 
} 

void FresnelAnalysisnk(Double_t n, Double_t alpha, Double_t d,
            Double_t lambda, Double_t Tm, Double_t Rm, nacon* nac) {

    // some printing out and value precision stuff
    //cout.setf(ios::fixed | ios::showpoint);
    //cout.precision(5);

    Double_t k = (lambda*alpha)/(4*PI);

    cout << endl << "---------------------------------------" << endl;
    cout << "\nStarting using 2D Newton method for n and k...\n" << endl;
    cout<<"  loop       n       k       TFunc       RFunc         dx\
        dy" << endl;

    if(RunNewtonTwoD(n,k,g_MAXLOOP,g_ACCURACY,Tm,Rm,d,lambda)) {
        cout<< " \n successfully finding the root!\n" << endl;
        alpha = (k*4.0*PI)/(lambda);
        PrintTandR(n,alpha,d,lambda);
    } else {
        cout << "\nfailed to find root " << endl;
    }
    cout << endl << "---------------------------------------" << endl;

    nac->n = n;
    nac->alpha = alpha;

}
////////////////////////////////////////////////////////////////////////////
//////// Newton method /////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////

// 1D Newton method with fixed n value for testing 
Int_t RunNewtonOneD(Double_t &n, Double_t &k, Int_t maxLoop, 
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
        dk = - Tmf/Tmk; 
        k = k + dk; 
        cout<<setw(3)<<i<<setw(12)<<n<<setw(12)<<k<<setw(12)<<setw(12) 
            <<Tmf<<setw(12)<<Rmf<<setw(12)<<dn<<setw(12)<<dk<<endl; 
        } 
    while(fabs(dk) > accuracy && (--maxLoop)); 
 
    // return the maxLoop as the flag to express finding 
    // a root successfully 
    return maxLoop; 
 
}

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
        }
    while((fabs(dn) > accuracy || fabs(dk) > accuracy) && (--maxLoop));

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
////////////////////////////////////////////////////////////////////////////
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
////////////////////////////////////////////////////////////////////////////
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

/////////////////////////////////////////////////////////////////////////////
/////////// Print T and R info //////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////
void PrintTandR(Double_t n, Double_t alpha, Double_t d, Double_t lambda){

    //lambda = lambda/1.0e-6; // unit: mm->nm
    //alpha = alpha/0.1; // unit: mm-1 -> cm-1
    Double_t k = (lambda*alpha)/(4*PI);

    Double_t FinalTmc = GetOpticalModelTValue(GetIT(k,d,lambda),
                            GetFR(n,k));
    Double_t FinalRmc = GetOpticalModelRValue(FinalTmc,
                            GetIT(k,d,lambda),GetFR(n,k));

    cout << "  d     wl(nm)          n         k          alpha(cm-1)\
          T         R" << endl;
    cout <<setw(3)<<d<<setw(8)<<lambda/1.0e-6<<setw(12)<<n<<setw(12)<<k<<setw(12)<<alpha/0.1
            <<FinalTmc <<setw(12)<<FinalRmc<<endl<< endl;


}

