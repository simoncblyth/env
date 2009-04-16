/************************************************************************\

    File: findNKNewton.cpp
    Created by: Taihsiang
    Date: Apr., 06, 2009
    Description:
                C++ source codes to use Fresnel equation
                and Newton method to analyze transmittance
                and reflectance data and then to get a numerical
                solution of index of refraction n and extinction
                coefficient k.
    Reference:




    Codeing convention:
        Index of refraction: key word "IOR"
        Extinction coefficient: key word "EC"
        Passing sequence:
            transmittance, reflectance, thickness, wavelength,
            IOR,EC

\************************************************************************/

//#define UNITTESTONED
//#define UNITTESTTWOD

#include <cmath>
#include <cfloat>
#include <fstream>
#include <iostream>
#include <string>
#include "findNKNewton.h"

using namespace std;

FresnelData::FresnelData(void) {

}

FresnelData::~FresnelData(void) {

}

FresnelData::FresnelData(   string thinTransmittanceFilename, string thinReflectanceFilename,
                            string thickTransmittanceFilename, string thickReflectanceFilename) {

    resetAllPrivate();
    loadFromFile(   thinTransmittanceFilename,thinReflectanceFilename,
                    thickTransmittanceFilename,thickReflectanceFilename);

}

void FresnelData::resetAllPrivate() {

    for(int i=0;i<TOTALDATANO;i++) {
        thinTransmittance_[i] = -1;
        thinReflectance_[i] = -1;
        thickTransmittance_[i] = -1;
        thickReflectance_[i] = -1;
        wavelength_[i] = -1;
        indexOfRefraction_[i] = -1;
        extinctionCoefficient_[i] = -1;
        numericalStatus_[i] = -1;
    }
    thinThickness_ = -1;
    thickThickness_ = -1;
}

int FresnelData::loadFromFile(  string thinTransmittanceFilename, string thinReflectanceFilename,
                                string thickTransmittanceFilename, string thickReflectanceFilename) {

    cout << "Reading in ...... " << thinTransmittanceFilename << endl;
    cout << "Reading in ...... " << thinReflectanceFilename << endl;
    cout << "Reading in ...... " << thickTransmittanceFilename << endl;
    cout << "Reading in ...... " << thickReflectanceFilename << endl;
    ifstream finThinTransmittance(thinTransmittanceFilename.data());
    ifstream finThinReflectance(thinReflectanceFilename.data());
    ifstream finThickTransmittance(thickTransmittanceFilename.data());
    ifstream finThickReflectance(thickReflectanceFilename.data());

    // thinWT:thinWavelengthFromTransmittance
    // thinWR:thinWavelengthFromReflectance
    long double thinWT(0), thinWR(0), thickWT(0), thickWR(0);
    int effectiveTotalDataNo(0);
    for(int i=0;i<TOTALDATANO; i++) {
        finThinTransmittance >> thinWT;
        finThinReflectance >> thinWR;
        finThickTransmittance >> thickWT;
        finThickReflectance >> thickWR;
        if((thinWT == thinWR) && (thickWT == thickWR) && (thickWT == thickWR)) {
            finThinTransmittance >> thinTransmittance_[i];
            finThinReflectance >> thinReflectance_[i];
            finThickTransmittance >> thickTransmittance_[i];
            finThickReflectance >> thickReflectance_[i];
            // unit %->0.01 nm->mm
            thinTransmittance_[i] = thinTransmittance_[i]*0.01;
            thinReflectance_[i] = thinReflectance_[i]*0.01;
            thickTransmittance_[i] = thickTransmittance_[i]*0.01;
            thickReflectance_[i] = thickReflectance_[i]*0.01;
            wavelength_[i] = thinWT*1.0e-6;
            effectiveTotalDataNo++;
        }
    }
    effectiveTotalDataNo_ = effectiveTotalDataNo;
    if(effectiveTotalDataNo_ < 0 || effectiveTotalDataNo_ == 0 ) return EXIT_FAILURE;
    if (finThinTransmittance.fail() || finThinReflectance.fail() || finThickTransmittance.fail() || finThickReflectance.fail()) return NK_ERROR;
    finThinTransmittance.close();
    finThinReflectance.close();
    finThickTransmittance.close();
    finThickReflectance.close();


    // Debug
    // and try the 8 degrees reflection incident angle deviation
    //for(int i=0;i<0;i++) {
    //    thinReflectance_[i] = thinReflectance_[i] - 0.01;
    //    thickReflectance_[i] = thickReflectance_[i] - 0.01;
    //}

    return NK_SUCCESS;

}

void FresnelData::dump(int dataNo) {

    if(numericalStatus_[dataNo] == NK_SUCCESS) {
        cout << 1000000.0*wavelength_[dataNo] << "nm\t" << indexOfRefraction_[dataNo]
        << "\t" << extinctionCoefficient_[dataNo]
        << "\tSUCCESS!!\t"
        << " " << evalTransmittanceConstrain(dataNo) << "\t" << evalReflectanceConstrain(dataNo)
        << endl;
    } else if(numericalStatus_[dataNo] == NK_ERROR) {
        //cout << 1000000.0*wavelength_[dataNo] << "nm\tN/A\tN/A\tFAILED!!\t"
        cout << 1000000.0*wavelength_[dataNo] << "nm\t" << indexOfRefraction_[dataNo]
        << "\t" << extinctionCoefficient_[dataNo] << "\tFAILED!!\t"
        << evalTransmittanceConstrain(dataNo) << "\t" << evalReflectanceConstrain(dataNo) << endl;
    } else {
        //cout << 1000000.0*wavelength_[dataNo] << "nm\tN/A\tN/A\tERROR/UNKNOWN\t"
        cout << 1000000.0*wavelength_[dataNo] << "nm\t" << indexOfRefraction_[dataNo]
        << "\t" << extinctionCoefficient_[dataNo] << "\tFAILED/UNKNOEN!!\t"
        << evalTransmittanceConstrain(dataNo) << "\t" << evalReflectanceConstrain(dataNo) << endl;
    }


}

void FresnelData::dumpToFile(string outputFilename) {

    ofstream fout;
    fout.open(outputFilename.data());

    for(int dataNo=0;dataNo<TOTALDATANO;dataNo++) {
        fout << 1000000.0*wavelength_[dataNo] << " " << indexOfRefraction_[dataNo]
        << " " << extinctionCoefficient_[dataNo] << " "
        << evalTransmittanceConstrain(dataNo) << " " << evalReflectanceConstrain(dataNo) << endl;
    }

}

int FresnelData::dumpSingleWavelengthNK(long double wavelengthValue) {

    int dataNo(-1);
    for(int i=0;i<TOTALDATANO;i++) {
        // Debug.
        //cout << wavelength_[i] << " " << wavelengthValue << endl;
        if(wavelength_[i] == wavelengthValue) dataNo = i;
    }

    if(dataNo == -1 && dataNo < -1) {
        cout << "No such wavelength value, STOP!" << endl;
        return EXIT_FAILURE;
    }

    if(numericalStatus_[dataNo] == NK_SUCCESS) {
        cout << wavelength_[dataNo] << "nm\t" << indexOfRefraction_[dataNo]
        << "\t" << extinctionCoefficient_[dataNo]
        << "\tSUCCESS!!" << endl;
    } else if(numericalStatus_[dataNo] == NK_ERROR) {
        cout << wavelength_[dataNo] << "\tFAILED!!" << endl;
    } else {
        cout << wavelength_[dataNo] << "\tERROR/UNKNOWN" << endl;
    }

    return NK_SUCCESS;

}

void FresnelData::set(long double *indexOfRefraction, long double *extinctionCoefficient) {

    for(int i=0;i<TOTALDATANO;i++) {
        indexOfRefraction_[i] = indexOfRefraction[i];
        extinctionCoefficient_[i] = extinctionCoefficient[i];
    }

}

void FresnelData::setKToRetry(int dataNo, int loopNo) {

    //if(dataNo == 12) cout << "void FresnelData::setKToRetry(int dataNo, int loopNo) is called" << endl;
    long double range = MAXEC - MINEC;
    long double ratio = (long double)(loopNo)/(long double)(MAXLOOP);
    extinctionCoefficient_[dataNo] = MINEC + range*ratio;

}

void FresnelData::setNKToRetry(int dataNo, int loopNo) {

    //cout << "void FresnelData::setNKToRetry(int dataNo, int loopNo)" << endl;
    long double rangeIOR = MAXIOR - MINIOR;
    long double rangeEC = MAXEC - MINEC;
    long double ratio = (long double)(loopNo)/(long double)(MAXLOOP);
    indexOfRefraction_[dataNo] = MINIOR + ratio*rangeIOR;
    extinctionCoefficient_[dataNo] = MINEC + ratio*rangeEC;

}

void FresnelData::setInitialParas(  long double indexOfRefraction, long double extinctionCoefficient,
                                    long double thinThickness, long double thickThickness) {

    for(int i=0;i<TOTALDATANO;i++) {
        indexOfRefraction_[i] = indexOfRefraction;
        extinctionCoefficient_[i] = extinctionCoefficient;
    }
    thinThickness_ = thinThickness;
    thickThickness_ = thickThickness;

}

int FresnelData::setSecondInitialParas(void) {

    int successCounter(0);
    for(int dataNo=0;dataNo<TOTALDATANO;dataNo++) {
        //if (numericalStatus_[data] == 0) { return EXIT_FAILURE;}
        if((numericalStatus_[dataNo] != NK_SUCCESS) && numericalStatus_[dataNo-1] == NK_SUCCESS && (dataNo !=0)) {
            indexOfRefraction_[dataNo] = indexOfRefraction_[dataNo-1];
            extinctionCoefficient_[dataNo] = extinctionCoefficient_[dataNo-1];
            numericalStatus_[dataNo] = NK_SUCCESS; // reactive the status for next one using
            successCounter++;
            // Debug
            //cout << wavelength_[dataNo] << " " << indexOfRefraction_[dataNo] << " " << extinctionCoefficient_[dataNo] << endl;
        }

        if(dataNo < 400) {
            if((numericalStatus_[dataNo-1] == NK_SUCCESS) && (((fabs(indexOfRefraction_[dataNo]/indexOfRefraction_[dataNo-1])) < 0.8) || ((fabs(indexOfRefraction_[dataNo]/indexOfRefraction_[dataNo-1])) > 1.2))){
                indexOfRefraction_[dataNo] = indexOfRefraction_[dataNo-1];
            }
        } else {
            if((numericalStatus_[dataNo-1] == NK_SUCCESS) && (((fabs(extinctionCoefficient_[dataNo]/extinctionCoefficient_[dataNo-1])) < 0.08) || ((fabs(extinctionCoefficient_[dataNo]/extinctionCoefficient_[dataNo-1])) > 8.0))) {
            extinctionCoefficient_[dataNo] = extinctionCoefficient_[dataNo-1];
            }
        }
    }

    // initialize numerical status again
    for(int dataNo=0;dataNo<TOTALDATANO;dataNo++) {
        numericalStatus_[dataNo] = -1;
    }


    if(successCounter == 0) {
        return EXIT_FAILURE;
    } else {
        return NK_SUCCESS;
    }

}

void FresnelData::get( long double *indexOfRefraction, long double *extinctionCoefficient) {

    for(int i=0;i<TOTALDATANO;i++) {
        indexOfRefraction[i] = indexOfRefraction_[i];
        extinctionCoefficient[i] = extinctionCoefficient_[i];
    }

}

long double FresnelData::evalInternalTransmittance(int dataNo) {

    AlphaTok(dataNo);

    long double internalTransmittance(-1);
    // How do I choose thickness?
    internalTransmittance = exp((-4.0*M_PI*extinctionCoefficient_[dataNo]*thickness_)/wavelength_[dataNo]);

    return internalTransmittance;

}

long double FresnelData::evalFrontSurfacePrimaryRelectance(int dataNo) {

    AlphaTok(dataNo);

    long double frontSurfacePrimaryRelectance(-1);
    long double numerator(0);
    long double denominator(0);
    numerator = (indexOfRefraction_[dataNo] - 1.0)*(indexOfRefraction_[dataNo] - 1.0)
                + extinctionCoefficient_[dataNo]*extinctionCoefficient_[dataNo];
    denominator =   (indexOfRefraction_[dataNo] + 1.0)*(indexOfRefraction_[dataNo] + 1.0)
                    + extinctionCoefficient_[dataNo]*extinctionCoefficient_[dataNo];
    frontSurfacePrimaryRelectance = numerator/denominator;

    // Debug.
    //cout << "FrontSurfacePrimaryRelectance " << frontSurfacePrimaryRelectance << endl;

    return frontSurfacePrimaryRelectance;

}

long double FresnelData::evalCaculatedGrossTransmittance(int dataNo) {

    long double caculatedGrossTransmittance(-1);
    long double numerator(0);
    long double denominator(0);
    numerator = (1.0 - evalFrontSurfacePrimaryRelectance(dataNo))*(1.0 - evalFrontSurfacePrimaryRelectance(dataNo))*evalInternalTransmittance(dataNo);
    denominator = 1.0 - evalFrontSurfacePrimaryRelectance(dataNo)*evalFrontSurfacePrimaryRelectance(dataNo)*evalInternalTransmittance(dataNo)*evalInternalTransmittance(dataNo);
    caculatedGrossTransmittance = numerator/denominator;

    return caculatedGrossTransmittance;

}

long double FresnelData::evalCaculatedGrossReflectance(int dataNo) {

    long double caculatedGrossReflectance(-1);
    caculatedGrossReflectance = evalFrontSurfacePrimaryRelectance(dataNo)*(1 + evalInternalTransmittance(dataNo)*evalCaculatedGrossTransmittance(dataNo));

    return caculatedGrossReflectance;

}

void FresnelData::newtonMethodRTRTT(int dataNo) {

    //cout << "Starting Newton method..." ;

    int maxLoop = MAXLOOP;
    do
    {
        //cout << "Loop..." << maxLoop << " ";
        if(maxLoop != MAXLOOP) setNKToRetry(dataNo, maxLoop);
        newtonMethodTwoDForNK(dataNo);
        //if(maxLoop%MAXLOOP == 0) cout << "RT...";
        if(numericalStatus_[dataNo] == NK_SUCCESS) {
            //cout << "RT is successful, try now RTT'"<< endl;
            newtonMethodOneDForK(dataNo);
            //if(maxLoop%MAXLOOP == 0) cout << "RTT..." ;
            int maxLoopTmp = MAXLOOP + 1;
            while(numericalStatus_[dataNo] != NK_SUCCESS && (--maxLoopTmp))
            {
                //cout << "maxLoop tmp " << maxLoopTmp << endl;
                setKToRetry(dataNo, maxLoopTmp);
                newtonMethodOneDForK(dataNo);
                //if(maxLoopTmp%MAXLOOP == 0) cout << "Retry RTT...";
            }
        } else {
            numericalStatus_[dataNo] = NK_ERROR;
        }
    }
    while(numericalStatus_[dataNo] != NK_SUCCESS && (--maxLoop));

    //cout << "Newton method finish!!!" << endl;

}

int FresnelData::newtonMethodRTRTTSingleWavelength(long double wavelengthValue) {

    //cout << "Run for single wavelength Newton method...." << endl;

    int dataNo(-1);
    for(int i=0;i<TOTALDATANO;i++) {
        // Debug.
        //cout << wavelength_[i] << " " << wavelengthValue << endl;
        if(wavelength_[i] == wavelengthValue) dataNo = i;
    }

    if(dataNo == -1 && dataNo < -1) {
        cout << "No such wavelength value for Newton method, STOP!" << endl;
        return EXIT_FAILURE;
    } else {
        newtonMethodRTRTT(dataNo);
    }

    return NK_SUCCESS;

}

void FresnelData::newtonMethodOneDForK(int dataNo) {

    //cout << endl << "FresnelData::newtonMethodOneDForK(int dataNo)" << endl;

#ifdef UNITTESTONED
        setNewtonParasForOneDUnitTest(dataNo);
#endif

    int maxLoop = MAXLOOP;
    do
    {
        if(setNewtonParas(dataNo,1,1) == NK_SUCCESS) {

            //cout << "Hello" << endl;
            //cout << maxLoop << endl;
            if(newtonParas_[3] != 0) {
    
                // Debug.
                //cout << "newtonParas_[0] " << newtonParas_[0] << " ,newtonParas_[3] " << newtonParas_[3];
                //cout << "newtonParas_[0]/newtonParas_[3] " << newtonParas_[0]/newtonParas_[3] << endl;
                newtonParas_[6] = newtonParas_[0]/newtonParas_[3];
                extinctionCoefficient_[dataNo] -= newtonParas_[6];


                // Debug.
                //if(dataNo == 12) {
                //cout << maxLoop << " " << indexOfRefraction_[dataNo] << " " << newtonParas_[6] << " " << extinctionCoefficient_[dataNo] << " " << newtonParas_[7] << " " << evalTransmittanceConstrain(dataNo) << " " << evalReflectanceConstrain(dataNo) << endl;
                //}



            } else {
                //cout << "newtonParas_[3] is 0" << endl;
                maxLoop = NK_ERROR; // forcing to stop
                numericalStatus_[dataNo] = NK_ERROR;
            }
        } else {
            //cout << "ASKING maxLoop to be 0" << endl;
            maxLoop = NK_ERROR;
            numericalStatus_[dataNo] = NK_ERROR;
        }
    }
    while((fabs(newtonParas_[7]) > KACCURACY || (evalTransmittanceConstrain(dataNo) > ACCURACY) || (evalReflectanceConstrain(dataNo) > ACCURACY)) && (--maxLoop));

    // Debug
    //cout << "dataNo is " << dataNo << endl;

    if((maxLoop > 0) && (validateValue(dataNo) == NK_SUCCESS)) {
        numericalStatus_[dataNo] = NK_SUCCESS;
        //cout << "FresnelData::newtonMethodOneDForK success!!" << endl;
    } else {
        numericalStatus_[dataNo] = NK_ERROR;
        //cout << "FresnelData::newtonMethodOneDForK failed!!" << endl;
    }

}

void FresnelData::newtonMethodTwoDForNK(int dataNo) {

    //cout << "FresnelData::newtonMethodTwoDForNK(int dataNo)" << endl;

#ifdef UNITTESTTWOD
    // Debug. Unit test
    setNewtonParasForTwoDUnitTest(dataNo);
#endif

    int maxLoop = MAXLOOP;
    long double delta(0), tmpDeltaIOR(0), tmpDeltaEC(0);
    do
    {
        if(setNewtonParas(dataNo,0,2) == NK_SUCCESS) {
            evalJacobian(dataNo);
            delta = newtonParas_[2]*newtonParas_[5] - newtonParas_[3]*newtonParas_[4];
            if(delta == 0) maxLoop = NK_ERROR; // forcing to stop
            //tmpDeltaIOR = (newtonParas_[3]*newtonParas_[1] - newtonParas_[0]*newtonParas_[5])/delta;
            //tmpDeltaEC = (newtonParas_[0]*newtonParas_[4] - newtonParas_[2]*newtonParas_[1])/delta;
            //if(dataNo == 12) {
            //    cout << "Loop\tn\tdn\tk\tdk\tdT\tdR" << endl;
            //    cout << maxLoop << "\t" << indexOfRefraction_[dataNo] << " " << newtonParas_[6] << " " << extinctionCoefficient_[dataNo] << " " << newtonParas_[7] << " " << evalTransmittanceConstrain(dataNo)  << " " << evalReflectanceConstrain(dataNo) << endl;
            //}
            //cout << "shift n\tshitf k" << endl;
            //cout << tmpDeltaIOR << "\t" << tmpDeltaEC << endl;
        } else {
            maxLoop = NK_ERROR;
        }
    }
    while(((fabs(newtonParas_[6]) > NACCURACY) || (fabs(newtonParas_[7]) > KACCURACY) || (evalTransmittanceConstrain(dataNo) > ACCURACY) || (evalReflectanceConstrain(dataNo) > ACCURACY)) && (--maxLoop));

    //cout << endl;

    if(maxLoop > 0) {
        numericalStatus_[dataNo] = NK_SUCCESS;
        //cout << "FresnelData::newtonMethodTwoDForNK success!!" << endl;
        //cout << "n is " << indexOfRefraction_[dataNo] << " ,k is " << extinctionCoefficient_[dataNo] << endl;
    } else {
        numericalStatus_[dataNo] = NK_ERROR;
        //cout << "FresnelData::newtonMethodTwoDForNK failed!!" << endl;
    }

}

// 0 for thin sample(RT), 1 for thick sample(RTT')
int FresnelData::setNewtonParas(int dataNo, int thickFlag, int methodDimensionFlag) {

    // Debug.
    //cout << "void FresnelData::setNewtonParas(int dataNo, int thickFlag)" << endl;

    if(methodDimensionFlag != 1 && methodDimensionFlag != 2) {
        cout << "Wrong methodDimensionFlag" << endl;
        return EXIT_FAILURE;
    }

    // TODO
    // need to be optimized.
    if(thickFlag == 0) {
        thickness_ = thinThickness_;
        for(int i=0;i<TOTALDATANO;i++) {
            transmittance_[i] = thinTransmittance_[i];
            reflectance_[i] = thinReflectance_[i];
        }
    } else if(thickFlag == 1) {
        thickness_ = thickThickness_;
        for(int i=0;i<TOTALDATANO;i++) {
            transmittance_[i] = thickTransmittance_[i];
            reflectance_[i] = thickReflectance_[i];
        }
    } else {
        cout << "Wrong thick flag!" <<  endl;
        return EXIT_FAILURE;
    }

    for(int i=0;i<6;i++) newtonParas_[i] = 0; // reset/initialize

    long double tmpIOR = indexOfRefraction_[dataNo];
    long double tmpEC = extinctionCoefficient_[dataNo];
    long double atmpFuncValue(0);
    long double btmpFuncValue(0);

#ifndef UNITTESTONED
#ifndef UNITTESTTWOD

    if(validateNumericalRange(dataNo) == NK_SUCCESS) {

        // Debug.
        //cout << "Caculating T partial k" << endl;

        newtonParas_[0] = evalTransmittanceConstrain(dataNo);
        newtonParas_[1] = evalReflectanceConstrain(dataNo);
 
        // T partial k
        extinctionCoefficient_[dataNo] += DELTAEC;
        atmpFuncValue = evalTransmittanceConstrain(dataNo);
        extinctionCoefficient_[dataNo] = tmpEC;
        extinctionCoefficient_[dataNo] -= DELTAEC;
        btmpFuncValue = evalTransmittanceConstrain(dataNo);
        extinctionCoefficient_[dataNo] = tmpEC;
        newtonParas_[3] = (atmpFuncValue - btmpFuncValue)/ (2*DELTAEC);
        //cout << "atmpFuncValue " << atmpFuncValue << endl;
        //cout << "btmpFuncValue " << btmpFuncValue << endl;
        //cout << "atmpFuncValue - btmpFuncValue = " << atmpFuncValue - btmpFuncValue << endl;
     
        if(methodDimensionFlag == 2) {
        // T partial n
        indexOfRefraction_[dataNo] += DELTA;
        atmpFuncValue = evalTransmittanceConstrain(dataNo);
        indexOfRefraction_[dataNo] = tmpIOR;
        indexOfRefraction_[dataNo] -= DELTA;
        btmpFuncValue = evalTransmittanceConstrain(dataNo);
        indexOfRefraction_[dataNo] = tmpIOR;
        newtonParas_[2] = (atmpFuncValue - btmpFuncValue)/ (2*DELTA);
        //cout << "atmpFuncValue " << atmpFuncValue << endl;
        //cout << "btmpFuncValue " << btmpFuncValue << endl;
        //cout << "atmpFuncValue - btmpFuncValue = " << atmpFuncValue - btmpFuncValue << endl;

        // R partial n
        indexOfRefraction_[dataNo] += DELTA;
        atmpFuncValue = evalReflectanceConstrain(dataNo);
        indexOfRefraction_[dataNo] = tmpIOR;
        indexOfRefraction_[dataNo] -= DELTA;
        btmpFuncValue = evalReflectanceConstrain(dataNo);
        indexOfRefraction_[dataNo] = tmpIOR;
        newtonParas_[4] = (atmpFuncValue - btmpFuncValue)/ (2*DELTA);
        //cout << "atmpFuncValue " << atmpFuncValue << endl;
        //cout << "btmpFuncValue " << btmpFuncValue << endl;
        //cout << "atmpFuncValue - btmpFuncValue = " << atmpFuncValue - btmpFuncValue << endl;
     
        // R partial k
        extinctionCoefficient_[dataNo] += DELTAEC;
        atmpFuncValue = evalReflectanceConstrain(dataNo);
        extinctionCoefficient_[dataNo] = tmpEC;
        extinctionCoefficient_[dataNo] -= DELTAEC;
        btmpFuncValue = evalReflectanceConstrain(dataNo);
        extinctionCoefficient_[dataNo] = tmpEC;
        newtonParas_[5] = (atmpFuncValue - btmpFuncValue)/ (2*DELTAEC);
        //cout << "atmpFuncValue " << atmpFuncValue << endl;
        //cout << "btmpFuncValue " << btmpFuncValue << endl;
        //cout << "atmpFuncValue - btmpFuncValue = " << atmpFuncValue - btmpFuncValue << endl;
        }
        return NK_SUCCESS;
    } else {
        numericalStatus_[dataNo] = NK_ERROR;
        //cout << "setNewtonParas Error!" << endl;
        return NK_ERROR;
    }


#endif
#endif

#ifdef UNITTESTONED

    /******************************************************\

        Unit Test for 1D Newton method

    \******************************************************/

    cout << "setNewtonParas for 1D" << endl;

    newtonParas_[0] = evalOneDUnitTestFormula(dataNo);

    // T partial k
    extinctionCoefficient_[dataNo] += DELTAEC;
    atmpFuncValue = evalOneDUnitTestFormula(dataNo);
    extinctionCoefficient_[dataNo] = tmpEC;
    extinctionCoefficient_[dataNo] -= DELTAEC;
    btmpFuncValue = evalOneDUnitTestFormula(dataNo);
    extinctionCoefficient_[dataNo] = tmpEC;
    newtonParas_[3] = (atmpFuncValue - btmpFuncValue)/ (2*DELTAEC);
    //cout << "atmpFuncValue " << atmpFuncValue << endl;
    //cout << "btmpFuncValue " << btmpFuncValue << endl;
    //cout << "atmpFuncValue - btmpFuncValue = " << atmpFuncValue - btmpFuncValue << endl;

    dumpOneDUnitTest(dataNo);

    return NK_SUCCESS;
#endif

#ifdef UNITTESTTWOD

    /******************************************************\

        Unit Test for 2D Newton method

    \******************************************************/

    newtonParas_[0] = evalTwoDUnitTestFormulaOne(dataNo);
    newtonParas_[1] = evalTwoDUnitTestFormulaTwo(dataNo);

    // T partial n
    indexOfRefraction_[dataNo] += DELTA;
    atmpFuncValue = evalTwoDUnitTestFormulaOne(dataNo);
    indexOfRefraction_[dataNo] = tmpIOR;
    indexOfRefraction_[dataNo] -= DELTA;
    btmpFuncValue = evalTwoDUnitTestFormulaOne(dataNo);
    indexOfRefraction_[dataNo] = tmpIOR;
    //cout << "atmpFuncValue " << atmpFuncValue << endl;
    //cout << "btmpFuncValue " << btmpFuncValue << endl;
    //cout << "atmpFuncValue - btmpFuncValue = " << atmpFuncValue - btmpFuncValue << endl;
    newtonParas_[2] = (atmpFuncValue - btmpFuncValue)/ (2*DELTA);

    // T partial k
    extinctionCoefficient_[dataNo] += DELTAEC;
    atmpFuncValue = evalTwoDUnitTestFormulaOne(dataNo);
    extinctionCoefficient_[dataNo] = tmpEC;
    extinctionCoefficient_[dataNo] -= DELTAEC;
    btmpFuncValue = evalTwoDUnitTestFormulaOne(dataNo);
    extinctionCoefficient_[dataNo] = tmpEC;
    newtonParas_[3] = (atmpFuncValue - btmpFuncValue)/ (2*DELTAEC);
    //cout << "atmpFuncValue " << atmpFuncValue << endl;
    //cout << "btmpFuncValue " << btmpFuncValue << endl;
    //cout << "atmpFuncValue - btmpFuncValue = " << atmpFuncValue - btmpFuncValue << endl;

    // R partial n
    indexOfRefraction_[dataNo] += DELTA;
    atmpFuncValue = evalTwoDUnitTestFormulaTwo(dataNo);
    indexOfRefraction_[dataNo] = tmpIOR;
    indexOfRefraction_[dataNo] -= DELTA;
    btmpFuncValue = evalTwoDUnitTestFormulaTwo(dataNo);
    indexOfRefraction_[dataNo] = tmpIOR;
    newtonParas_[4] = (atmpFuncValue - btmpFuncValue)/ (2*DELTA);
    //cout << "atmpFuncValue " << atmpFuncValue << endl;
    //cout << "btmpFuncValue " << btmpFuncValue << endl;
    //cout << "atmpFuncValue - btmpFuncValue = " << atmpFuncValue - btmpFuncValue << endl;

    // R partial k
    extinctionCoefficient_[dataNo] += DELTAEC;
    atmpFuncValue = evalTwoDUnitTestFormulaTwo(dataNo);
    extinctionCoefficient_[dataNo] = tmpEC;
    extinctionCoefficient_[dataNo] -= DELTAEC;
    btmpFuncValue = evalTwoDUnitTestFormulaTwo(dataNo);
    extinctionCoefficient_[dataNo] = tmpEC;
    newtonParas_[5] = (atmpFuncValue - btmpFuncValue)/ (2*DELTAEC);
    //cout << "atmpFuncValue " << atmpFuncValue << endl;
    //cout << "btmpFuncValue " << btmpFuncValue << endl;
    //cout << "atmpFuncValue - btmpFuncValue = " << atmpFuncValue - btmpFuncValue << endl;

    dumpTwoDUnitTest(dataNo);

    return NK_SUCCESS;
#endif


    // Debug.
    //for(int i=0;i<6;i++) cout << newtonParas_[i] << "\t";
    //cout << endl;


}

// validate value physically legal or not
int FresnelData::validateValue(int dataNo) {

    // different wavelength should has different validation...phsics
    if(dataNo < 400) {
        if((indexOfRefraction_[dataNo] > 1.0) && (indexOfRefraction_[dataNo] < 2.0) && ( extinctionCoefficient_[dataNo] > 0.0) && (extinctionCoefficient_[dataNo] < 1.0e-7)) {
            return NK_SUCCESS;
        } else {
            return NK_ERROR;
        }
    } else {
        if((indexOfRefraction_[dataNo] > 1.0) && (indexOfRefraction_[dataNo] < 2.0) && ( extinctionCoefficient_[dataNo] > 0.0) && (extinctionCoefficient_[dataNo] < 1.0e-4)) {
            return NK_SUCCESS;
        } else {
            return NK_ERROR;
        }
    }
}

// validate value not to be nan or inf
// 0 for o.k. 1 for invalid
int FresnelData::validateNumericalRange(int dataNo) {

    //cout << "Validation of parameters ";
    //if(dataNo < 30) cout << "n is " << indexOfRefraction_[dataNo] << " ,k is " << extinctionCoefficient_[dataNo] << endl;
    if(fabs(indexOfRefraction_[dataNo]) > 10.0 || fabs(extinctionCoefficient_[dataNo]) > 1.0) {
        //cout << "Values are invalid! " << endl;
        return NK_ERROR;
    } else {
        //cout << "values are valid " << endl;
        return NK_SUCCESS;
    }

}

void FresnelData::setECToAlpha(int dataNo) {

    cout << "Before FresnelData::setECToAlpha(int dataNo), EC is " << extinctionCoefficient_[dataNo] << endl;
    cout << "wavelenhth is " << wavelength_[dataNo] << endl;
    extinctionCoefficient_[dataNo] = (extinctionCoefficient_[dataNo]*(4.0)*M_PI)/wavelength_[dataNo];
    cout << "Alpha " << extinctionCoefficient_[dataNo] << endl;

}

void FresnelData::setAlphaToEC(int dataNo) {

    extinctionCoefficient_[dataNo] = wavelength_[dataNo]*extinctionCoefficient_[dataNo]/((4.0)*M_PI);

}

void FresnelData::evalJacobian(int dataNo) {

    long double delta = newtonParas_[2]*newtonParas_[5] - newtonParas_[3]*newtonParas_[4];
    if(delta == 0) {
        numericalStatus_[dataNo] = NK_ERROR;
    } else {
        // Debug. Please check again to avoid keying in wrong
        newtonParas_[6] = (newtonParas_[3]*newtonParas_[1] - newtonParas_[0]*newtonParas_[5])/delta;
        indexOfRefraction_[dataNo] += newtonParas_[6];
        //cout << "indexOfRefraction_[dataNo] " << indexOfRefraction_[dataNo] << endl;
        newtonParas_[7] = (newtonParas_[0]*newtonParas_[4] - newtonParas_[2]*newtonParas_[1])/delta;
        extinctionCoefficient_[dataNo] += newtonParas_[7];
        //cout << "extinctionCoefficient_[dataNo] " << extinctionCoefficient_[dataNo] << endl;
        numericalStatus_[dataNo] = NK_SUCCESS;
    }

}


long double FresnelData::evalTransmittanceConstrain(int dataNo) {

    return evalCaculatedGrossTransmittance(dataNo) - transmittance_[dataNo];

}

long double FresnelData::evalReflectanceConstrain(int dataNo) {

    return evalCaculatedGrossReflectance(dataNo) - reflectance_[dataNo];

}

void FresnelData::kToAlpha(int dataNo){

    alpha_[dataNo] = wavelength_[dataNo]/(4.0*M_PI*extinctionCoefficient_[dataNo]);

}

void FresnelData::AlphaTok(int dataNo) {

    extinctionCoefficient_[dataNo] = (4.0*M_PI*alpha_[dataNo])/wavelength_[dataNo];

}

/*********************************************************\

    Unit Test

    Test 1D Newton mehotd with
    x = 7
    9x^3+78 = 3165

\*********************************************************/
/*
void FresnelData::setNewtonParasForOneDUnitTest(int dataNo) {

    extinctionCoefficient_[dataNo] = 8.3;

    dumpOneDUnitTest(dataNo);

}

long double FresnelData::evalOneDUnitTestFormula(int dataNo) {

    return 9.0*extinctionCoefficient_[dataNo]*extinctionCoefficient_[dataNo]*extinctionCoefficient_[dataNo] + 78.0 - 3165.0;

}

void FresnelData::dumpOneDUnitTest(int dataNo) {

    cout << "The parameters should be ......" << endl;
    cout << "f\tfx" << endl;
    cout << evalOneDUnitTestFormula(dataNo) << "\t"
    << 27.0*extinctionCoefficient_[dataNo]*extinctionCoefficient_[dataNo] << endl;

}
*/

/*********************************************************\

    Unit Test

    Test 2D Newton method with
    (x,y) = (6,7)
    3x^x + y = 115
    4x + 5y^3 = 1739

\*********************************************************/
/*
void FresnelData::setNewtonParasForTwoDUnitTest(int dataNo) {

    indexOfRefraction_[dataNo] = 5.0;
    extinctionCoefficient_[dataNo] = 8.3;

    dumpTwoDUnitTest(dataNo);

}

long double FresnelData::evalTwoDUnitTestFormulaOne(int dataNo) {

    return 3.0*indexOfRefraction_[dataNo]*indexOfRefraction_[dataNo] + extinctionCoefficient_[dataNo] - 115.0;

}

long double FresnelData::evalTwoDUnitTestFormulaTwo(int dataNo) {

    return 4.0*indexOfRefraction_[dataNo]+5.0*extinctionCoefficient_[dataNo]*extinctionCoefficient_[dataNo]*extinctionCoefficient_[dataNo] - 1739.0;

}

void FresnelData::dumpTwoDUnitTest(int dataNo) {

    cout << "The parameters should be ......" << endl;
    cout << "f\tg\tfx\tfy\tgx\tgy" << endl;
    cout << evalTwoDUnitTestFormulaOne(dataNo) << "\t" << evalTwoDUnitTestFormulaTwo(dataNo) << "\t"
    << 6.0*indexOfRefraction_[dataNo] << "\t1\t4\t"
    << 15.0*extinctionCoefficient_[dataNo]*extinctionCoefficient_[dataNo] << endl;

}
*/
