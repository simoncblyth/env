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
// // analyzing at single wavelength
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
// // analyzing a wavelength range
// shell prompt> root
// root cint> .L FresnelAnalysis.C
// root cint> main("1-1-1-1.csv","1-1-2-1.csv",
//                      "2-1-1-1.csv","2-1-2-1.csv",
//                      1.505,0.009,10.14,14.80)
//
// main(transmission data of the thin sample,
//      reflection data of the thin sample,
//      transmission data of the thick sample,
//      reflection data of the thick sample,
//      initial index of refraction value for Newton method,
//      initial alpha value for Newton method,
//      thin sample thickness,
//      thick sample thickness)
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
#define DELTA 1.0e-9
#define g_MAXLOOP 99
#define g_ACCURACY 1.0e-6
#define DATASIZE 601
#define ANANO 601

using namespace std;

// n and alpha value container
typedef struct nacon
{
    Double_t n;
    Double_t alpha;
    Double_t deltat;
    Double_t deltar;
    Int_t status; // flag to show process succesful(1) or not(0)
};



// read in data and analyzing it
//
// data format
//
// wl(nm) TorT
// 800.0  0.920
// 790.0  0.915
//

// the main
void main(TString tfile, TString rfile,
            TString thtfile, TString thrfile,
            Double_t n, Double_t alpha,
            Double_t thin,Double_t thick) {

    Double_t wldataContainer[DATASIZE]={0.0};
    Double_t tdataContainer[DATASIZE]={0.0};
    Double_t rdataContainer[DATASIZE]={0.0};
    Double_t thtdataContainer[DATASIZE]={0.0};
    Double_t thrdataContainer[DATASIZE]={0.0};
    cout << "Reading in data......." << endl;
    ReadData(tfile, wldataContainer, tdataContainer);
    ReadData(rfile, wldataContainer, rdataContainer);
    ReadData(thtfile, wldataContainer, thtdataContainer);
    ReadData(thrfile, wldataContainer, thrdataContainer);

    Double_t resultn[DATASIZE]={0.0};
    Double_t resultk[DATASIZE]={0.0};
    Double_t resultatt[DATASIZE]={0.0};
    Int_t resultstatus[DATASIZE];
    for(Int_t i=0;i<DATASIZE;i++) {
        resultstatus[i]=0;
        //cout << resultatt[i] << " ";
    }

    cout << "wl(nm)\tn\tk\tatt(m)\tLoops\tFind n and k?" << endl;

    //Fakedata(wldataContainer, tdataContainer, rdataContainer, thtdataContainer, thrdataContainer);
    //if(CheckDataFormate(tdataContainer,rdataContainer,thtdataContainer,thrdataContainer)==0) {
    if(2>1){
        for(Int_t i=0;i<ANANO;i++){
            cout << "The " << i << " th data" << endl;
            // avoid to deal with the data which is smaller than machine precision
            if(tdataContainer[i] < 0.0011 || thtdataContainer[i] < 0.0011) {
                resultstatus[i] =0;
            } else {
                if(SucApp(n, alpha, wldataContainer[i],
                    thin, tdataContainer[i], rdataContainer[i],
                    thick, thtdataContainer[i], thrdataContainer[i],
                    resultn[i], resultk[i], resultatt[i], resultstatus[i]) != 0 ) {
                    resultstatus[i] = 0;
                    }
            }
        }
    } else break;

    HistonandkFilter(DATASIZE,wldataContainer,resultn,resultk,resultatt,resultstatus);
    cout << "done!!";

}

//////////  check the input data format legal or not /////////////////////////////
void Fakedata(Double_t a[], Double_t b[],Double_t c[],Double_t d[],Double_t e[]) {
    for(Int_t i=0;i<ANANO;i++){
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
//////////////////////////////////////////////////////////////////////////////

////////// some array, data stream handling /////////////////////////////////
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
    const Int_t inputSize = DATASIZE;

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

}

// copy a array A to another array b
void CopyArray(Double_t a[], Double_t b[], Int_t size) {
    for(Int_t i=0;i<size;i++) {
        b[i] = a[i];
    }
} 
/////////////////////////////////////////////////////////////////



// successive approach
Int_t reSucApp(Double_t n, Double_t alpha, Double_t lambda,
            Double_t thin, Double_t thinTm, Double_t thinRm,
            Double_t thick, Double_t thickTm, Double_t thickRm,
            Double_t &resultn, Double_t &resultk, Double_t &resultatt, Int_t &resultstatus) {

    Int_t maxLoop = g_MAXLOOP*2;
    Int_t oddEven(0);
    do
    {
    SucApp(n, alpha, lambda,
            thin, thinTm, thinRm,
            thick, thickTm, thickRm,
            resultn, resultk, resultatt, resultstatus);
    if(oddEven%2 ==0) { n += n*0.01; } else { alpha += alpha*0.01;}
    } while(resultstatus!=1 && --maxLoop);

    if(resultstatus==1) {return 0;} else {return -1;}

}

Int_t SucApp(Double_t n, Double_t alpha, Double_t lambda,
            Double_t thin, Double_t thinTm, Double_t thinRm,
            Double_t thick, Double_t thickTm, Double_t thickRm,
            Double_t &resultn, Double_t &resultk, Double_t &resultatt, Int_t &resultstatus) {

    nacon nac;
    Int_t maxLoop = g_MAXLOOP;
    Double_t deltan(0);
    Double_t deltaalpha(0);
    Double_t thinn(0);
    Double_t thickn(0)
    Double_t deltaalpha(0);
    Double_t thinalpha(0);
    Double_t thickalpha(0);

    //cout << lambda;
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
/*
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
*/
        resultn = nac.n;

        resultk = ((nac.alpha)/(4.0*PI))*lambda; // unitless
        resultatt = (1.0/(nac.alpha))/1000.0; // unit mm -> meter

        }
    while((nac.n<1 || nac.n==1 || fabs(deltan) > 0.01 || nac.alpha<0 || nac.alpha==0 || fabs(deltaalpha) > 0.001) && (--maxLoop));
    // some k is really close to 0 but it's negative. regard it as 0 
    if((nac.alpha<0) && ((-1)*((nac.alpha)/(4.0*PI))*lambda < 1.0e-6)) {
        nac.alpha = (-1)*nac.alpha;
        Double_t smallk = (nac.alpha*lambda)/(4*PI);
        Double_t deltathinTm = thinTm - GetOpticalModelTValue(GetIT(smallk,thin,lambda),GetFR(n,smallk));
        Double_t deltathinRm = thinRm - GetOpticalModelRValue(thinTm,GetIT(smallk,thin,lambda),GetFR(n,smallk));
        Double_t deltathickTm = thickTm - GetOpticalModelTValue(GetIT(smallk,thick,lambda),GetFR(n,smallk));
        Double_t deltathickRm = thickRm - GetOpticalModelRValue(thickTm,GetIT(smallk,thick,lambda),GetFR(n,smallk));
        if(deltathinTm>0.001 || deltathinRm>0.01 || deltathickTm>0.001 || deltathickRm>0.01 || maxLoop) {
            // TODO.......how to skip this value??
            //nac.alpha = (4.*PI*1.0e-8)/lambda;
            nac.alpha = 1.;
            resultstatus = 0;
            //cout << " N/A       N/A     N/A     " << g_MAXLOOP - maxLoop + 1 << "\tFAILED!!!!!!! :(" << endl;
        }
    } else {
        cout << " " << resultn << "  " << resultk << "  " << resultatt << "   " << g_MAXLOOP - maxLoop + 1;
        if(maxLoop){        
            //cout << "\tSUCCESSESSFUL :)" << endl;
            resultstatus = 1;
        } else {
            //cout << "\tFAILED!!!!!!! :(" << endl;
            resultstatus = 0;
        }
    }
    //cout << "*****************************************************" << endl;
    //cout << "*****************************************************" << endl;
    //cout << "*****************************************************\n\n\n" << endl;

    return 0;
}

Int_t FresnelAnalysisk(Double_t n, Double_t alpha, Double_t d, 
            Double_t lambda, Double_t Tm, Double_t Rm, nacon* nac) { 

    Int_t status(-1); 
    // some printing out and value precision stuff 
    //cout.setf(ios::fixed | ios::showpoint); 
    //cout.precision(5); 

    Double_t k = (lambda*alpha)/(4*PI); 

    //cout << endl << "---------------------------------------" << endl;
    //cout << "\nStarting using 1D Newton method for k only...\n" << endl; 
    //cout<<"  loop       n       k       TFunc       RFunc         dx     dy" << endl; 
 
    if(RunNewtonOneD(n,k,g_MAXLOOP,g_ACCURACY,Tm,Rm,d,lambda)) { 
        //cout<< " \tsuccessful";
        alpha = (k*4.0*PI)/(lambda);
        //PrintTandR(n,alpha,d,lambda);
        status = 0;
    } else { 
        //cout << "\tfailed";
    } 
    //cout << endl << "---------------------------------------" << endl;

    nac->n = n;
    nac->alpha = alpha;

    return status;

} 

Int_t FresnelAnalysisnk(Double_t n, Double_t alpha, Double_t d,
            Double_t lambda, Double_t Tm, Double_t Rm, nacon* nac) {

    Int_t status(-1);
    // some printing out and value precision stuff
    //cout.setf(ios::fixed | ios::showpoint);
    //cout.precision(5);

    Double_t k = (lambda*alpha)/(4*PI);

    //cout << endl << "---------------------------------------" << endl;
    //cout << "\nStarting using 2D Newton method for n and k...\n" << endl;
    //cout<<"  loop       n       k       TFunc       RFunc         dx\
    //    dy" << endl;

    if(RunNewtonTwoD(n,k,g_MAXLOOP,g_ACCURACY,Tm,Rm,d,lambda)) {
        //cout<< " \tsuccessful";
        alpha = (k*4.0*PI)/(lambda);
        //PrintTandR(n,alpha,d,lambda);
        status = 0;
    } else {
        //cout << "\tfailed";
    }
    //cout << endl << "---------------------------------------" << endl;

    nac->n = n;
    nac->alpha = alpha;

    return status;
}
////////////////////////////////////////////////////////////////////////////
//////// Newton method /////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////

// 1D Newton method with fixed n value 
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
        if(Tmk==0 || k>1e20) {
            break; } else {
            dk = - Tmf/Tmk;
            if(lambda==277) ;//cout << Tmk;
        }
        k = k + dk; 
        //cout<<setw(3)<<i<<setw(12)<<n<<setw(12)<<k<<setw(12)<<setw(12) 
        //    <<Tmf<<setw(12)<<Rmf<<setw(12)<<dn<<setw(12)<<dk<<endl; 
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
        //cout<<setw(3)<<i<<setw(12)<<n<<setw(12)<<k<<setw(12)<<setw(12)
        //    <<Tmf<<setw(12)<<Rmf<<setw(12)<<dn<<setw(12)<<dk<<endl;
        }
    while((fabs(dn) > accuracy || fabs(dk) > accuracy) && (--maxLoop));
    //cout << newn << " " << newk << "YAYAYA" << endl;
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

// Bug!!!
Double_t RPartialk(Double_t n, Double_t k, Double_t d, Double_t lambda,
                Double_t Tmc, Double_t Rm) {

    Double_t y = (RFunc(n,k+DELTA/2,d,lambda,Tmc,Rm)
                -RFunc(n,k-DELTA/2,d,lambda,Tmc,Rm))/DELTA;

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
    if(del==0) del = g_ACCURACY;
    dn = (Tmk*Rmf - Tmf*Rmk)/del;
    // need Confirm... Debug.
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
/////////// Print and draw T and R info /////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////
void HistonandkFilter(Int_t size, Double_t wl[], Double_t n[], Double_t k[], Double_t att[], Int_t status[]) {

    Int_t successfulcount(0);
    Int_t fillingcount(0);

    for(Int_t i=0;i<size;i++) {
        if(status[i]==1) {
            successfulcount++;
            //cout << successfulcount << " ";
        }
    }

    const Int_t newsize = successfulcount;
    Double_t newwl[newsize]={0.};
    Double_t newn[newsize]={0.};
    Double_t newk[newsize]={0.};
    Double_t newatt[newsize]={0.};

    for(Int_t i=0;i<size;i++) {
        if(status[i]==1) {
            newwl[fillingcount] = wl[i];
            newn[fillingcount] = n[i];
            newk[fillingcount] = k[i];
            newatt[fillingcount] = att[i];
            //cout << fillingcount << "\t" << newwl[fillingcount] << "\t" << newn[fillingcount] << "\t" << newk[fillingcount] << endl;
            fillingcount++;
        }
    }

    Histonandk(successfulcount,newwl,newn,newk,newatt);

}

void Histonandk(Int_t size, Double_t wl[], Double_t n[], Double_t k[], Double_t att[]) {

    TCanvas *c1 = new TCanvas(
        "c1","Acrylic Optical Parameters",200,10,700,900);
//    title = new TPaveText(.2,0.96,.8,.995);
//    title->AddText("Optical Parameters");
//    title->Draw();
    c1->Divide(2,2);
/*
    pad1 = new TPad("pad1","Index of Refration",0.03,0.50,0.98,0.95,21);
    pad2 = new TPad("pad2","Extinction Coeffition",0.03,0.02,0.98,0.48,21);
    pad3 = new TPad("pad3","Attenuation Length",0.03,0.50,0.98,0.95,21);
    pad1->Draw();
    pad2->Draw();
    pad3->Draw();
*/
    //for(Int_t i=0;i<ANANO;i++) {
    //    cout << i << "\t" << n[i] << "\t" << k[i] << endl;
    //}


    c1->cd(1);
    grn = new TGraph(size, wl, n);
//    pad1->cd();
    //grn->SetLineColor(2);
    //grn->SetLineWidth(4);
    //grn->SetMarkerColor(4);
    //grn->SetMarkerStyle(21);
    grn->SetTitle("Index of Refraction V.S. Wavelength");
    grn->GetXaxis()->SetTitle("nm");
    grn->GetYaxis()->SetTitle("n");
    grn->Draw("A*");

    c1->cd(2);
    grk = new TGraph(size, wl, k);
//    pad2->cd();
    //grk->SetLineColor(2);
    //grk->SetLineWidth(4);
    //grk->SetMarkerColor(4);
    //grk->SetMarkerStyle(21);
    grk->SetTitle("Extinction Coeffition V.S. Wavelength");
    grk->GetXaxis()->SetTitle("nm");
    grk->GetYaxis()->SetTitle("k");
    grk->Draw("A*");

    c1->cd(3);
    gratt = new TGraph(size, wl, att);
//    pad3->cd();
    //gratt->SetLineColor(2);
    //gratt->SetLineWidth(4);
    //gratt->SetMarkerColor(4);
    //gratt->SetMarkerStyle(21);
    gratt->SetTitle("Attenuation Length V.S. Wavelength");
    gratt->GetXaxis()->SetTitle("nm");
    gratt->GetYaxis()->SetTitle("meter");
    gratt->Draw("A*");

}
//void Histon(TH1D *nvswl

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

