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
    double thinWT(0), thinWR(0), thickWT(0), thickWR(0);
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
    // Debug.
    cout << "FresnelData::loadFromFile() A" << endl;
    if (finThinTransmittance.fail() || finThinReflectance.fail() || finThickTransmittance.fail() || finThickReflectance.fail()) return NK_ERROR;
    finThinTransmittance.close();
    finThinReflectance.close();
    finThickTransmittance.close();
    finThickReflectance.close();

    return NK_SUCCESS;

}

void FresnelData::dump(int dataNo) {

    if(numericalStatus_[dataNo] == 0) {
        cout << wavelength_[dataNo] << "\t" << indexOfRefraction_[dataNo]
        << "\t" << extinctionCoefficient_[dataNo]
        << "\tSUCCESS!!" << endl;
    } else if(numericalStatus_[dataNo] == 1) {
        cout << wavelength_[dataNo] << "\tFAILED!!" << endl;
    } else {
        cout << wavelength_[dataNo] << "\tERROR/UNKNOWN" << endl;
    }


}

int FresnelData::dumpSingleWavelengthNK(double wavelengthValue) {

    int dataNo(-1);
    wavelengthValue = wavelengthValue * 1.0e-6; // uniform untis
    for(int i=0;i<TOTALDATANO;i++) {
        // Debug.
        cout << wavelength_[i] << " " << wavelengthValue << endl;
        if(wavelength_[i] == wavelengthValue) dataNo = i;
    }

    if(dataNo == -1 && dataNo < -1) {
        cout << "No such wavelength value, STOP!" << endl;
        return EXIT_FAILURE;
    }

    if(numericalStatus_[dataNo] == 0) {
        cout << wavelength_[dataNo] << "\t" << indexOfRefraction_[dataNo]
        << "\t" << extinctionCoefficient_[dataNo]
        << "\tSUCCESS!!" << endl;
    } else if(numericalStatus_[dataNo] == 1) {
        cout << wavelength_[dataNo] << "\tFAILED!!" << endl;
    } else {
        cout << wavelength_[dataNo] << "\tERROR/UNKNOWN" << endl;
    }

    return NK_SUCCESS;

}

void FresnelData::set(double *indexOfRefraction, double *extinctionCoefficient) {

    for(int i=0;i<TOTALDATANO;i++) {
        indexOfRefraction_[i] = indexOfRefraction[i];
        extinctionCoefficient_[i] = extinctionCoefficient[i];
    }

}

void FresnelData::setKToRetry(int dataNo, int loopNo) {

    double range = MAXEC - MINEC;
    double ratio = double(loopNo)/double(MAXLOOP);
    extinctionCoefficient_[dataNo] = MINEC + range*ratio;

}

void FresnelData::setNKToRetry(int dataNo, int loopNo) {

    double rangeIOR = MAXIOR - MINIOR;
    double rangeEC = MAXEC - MINEC;
    double ratio = double(loopNo)/double(MAXLOOP);
    indexOfRefraction_[dataNo] = MINIOR + ratio*rangeIOR;
    extinctionCoefficient_[dataNo] = MINEC + ratio*rangeEC;

}

void FresnelData::setInitialParas(  double indexOfRefraction, double extinctionCoefficient,
                                    double thinThickness, double thickThickness) {

    for(int i=0;i<TOTALDATANO;i++) {
        indexOfRefraction_[i] = indexOfRefraction;
        extinctionCoefficient_[i] = extinctionCoefficient;
    }
    thinThickness_ = thinThickness;
    thickThickness_ = thickThickness;

}

void FresnelData::get( double *indexOfRefraction, double *extinctionCoefficient) {

    for(int i=0;i<TOTALDATANO;i++) {
        indexOfRefraction[i] = indexOfRefraction_[i];
        extinctionCoefficient[i] = extinctionCoefficient_[i];
    }

}

double FresnelData::evalInternalTransmittance(int dataNo) {

    double internalTransmittance(-1);
    // How do I choose thickness?
    internalTransmittance = exp((-4.0*M_PI*extinctionCoefficient_[dataNo]*thickness_)/wavelength_[dataNo]);

    return internalTransmittance;

}

double FresnelData::evalFrontSurfacePrimaryRelectance(int dataNo) {

    double frontSurfacePrimaryRelectance(-1);
    double numerator(0);
    double denominator(0);
    numerator = (indexOfRefraction_[dataNo] - 1.0)*(indexOfRefraction_[dataNo] - 1.0)
                + extinctionCoefficient_[dataNo]*extinctionCoefficient_[dataNo];
    denominator =   (indexOfRefraction_[dataNo] + 1.0)*(indexOfRefraction_[dataNo] + 1.0)
                    + extinctionCoefficient_[dataNo]*extinctionCoefficient_[dataNo];
    frontSurfacePrimaryRelectance = numerator/denominator;

    return frontSurfacePrimaryRelectance;

}

double FresnelData::evalCaculatedGrossTransmittance(int dataNo) {

    double caculatedGrossTransmittance(-1);
    double numerator(0);
    double denominator(0);
    numerator = (1.0 - evalFrontSurfacePrimaryRelectance(dataNo))*(1.0 - evalFrontSurfacePrimaryRelectance(dataNo))*evalInternalTransmittance(dataNo);
    denominator = 1.0 - evalFrontSurfacePrimaryRelectance(dataNo)*evalFrontSurfacePrimaryRelectance(dataNo)*evalInternalTransmittance(dataNo)*evalInternalTransmittance(dataNo);
    caculatedGrossTransmittance = numerator/denominator;

    return caculatedGrossTransmittance;

}

double FresnelData::evalCaculatedGrossReflectance(int dataNo) {

    double caculatedGrossReflectance(-1);
    caculatedGrossReflectance = evalFrontSurfacePrimaryRelectance(dataNo)*(1 + evalInternalTransmittance(dataNo)*evalCaculatedGrossTransmittance(dataNo));

    return caculatedGrossReflectance;

}

void FresnelData::newtonMethodRTRTT(int dataNo) {

    cout << "Starting Newton method..." << endl;

    int maxLoop = MAXLOOP;
    do
    {
        newtonMethodTwoDForNK(dataNo);
        newtonMethodOneDForK(dataNo);
        // retry RTT' first
        while(numericalStatus_[dataNo]!=0 && (--maxLoop))
        {
            setKToRetry(dataNo, maxLoop);
            newtonMethodOneDForK(dataNo);
        }
        // if retrying RTT' failed, using the slower but larger range scanning
        while(numericalStatus_[dataNo]!=0 && (--maxLoop))
        {
            setNKToRetry(dataNo, maxLoop);
            newtonMethodTwoDForNK(dataNo);
            newtonMethodOneDForK(dataNo);
        }   
    }
    while(numericalStatus_[dataNo]!=0 && (--maxLoop));


}

int FresnelData::newtonMethodRTRTTSingleWavelength(double wavelengthValue) {

    cout << "Run for single wavelength Newton method...." << endl;

    int dataNo(-1);
    wavelengthValue = wavelengthValue * 1.0e-6; // uniform untis
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

    cout << "FresnelData::newtonMethodOneDForK(int dataNo)" << endl;

    int maxLoop = MAXLOOP;
    do
    {
        setNewtonParas(dataNo,1);
        if(newtonParas_[3] != 0) {
            extinctionCoefficient_[dataNo] += newtonParas_[0]/newtonParas_[3];
        } else {
            maxLoop = 0; // forcing to stop
            numericalStatus_[dataNo] = 1;
        }
        cout << maxLoop << endl;
    }
    //while(fabs(newtonParas_[0]/newtonParas_[3]) > ACCURACY && --maxLoop);
    while(--maxLoop);

    if(maxLoop > 0) {
        numericalStatus_[dataNo] = 0;
        cout << "FresnelData::newtonMethodOneDForK success!!" << endl;
    } else {
        numericalStatus_[dataNo] = 1;
        cout << "FresnelData::newtonMethodOneDForK failed!!" << endl;
    }

}

void FresnelData::newtonMethodTwoDForNK(int dataNo) {

    cout << "FresnelData::newtonMethodTwoDForNK(int dataNo)" << endl;

    int maxLoop = MAXLOOP;
    double delta(0), tmpDeltaIOR(0), tmpDeltaEC(0);
    do
    {
        setNewtonParas(dataNo,0);
        evalJacobian(dataNo);
        delta = newtonParas_[2]*newtonParas_[5] - newtonParas_[3]*newtonParas_[4];
        cout << "delta " << delta << endl;
        if(delta == 0) maxLoop = 0; // forcing to stop
        tmpDeltaIOR = (newtonParas_[3]*newtonParas_[1] - newtonParas_[0]*newtonParas_[5])/delta;
        tmpDeltaEC = (newtonParas_[0]*newtonParas_[4] - newtonParas_[2]*newtonParas_[1])/delta;
        cout << "Two D loop No " << maxLoop << endl;
    }
    while((fabs(tmpDeltaIOR) > ACCURACY) && (fabs(tmpDeltaEC)> ACCURACY) && (--maxLoop));

    if(maxLoop > 0) {
        numericalStatus_[dataNo] = 0;
        cout << "FresnelData::newtonMethodTwoDForNK success!!" << endl;
        cout << "n is " << indexOfRefraction_[dataNo] << " ,k is " << extinctionCoefficient_[dataNo] << endl;
    } else {
        numericalStatus_[dataNo] = 1;
        cout << "FresnelData::newtonMethodTwoDForNK failed!!" << endl;
    }

}

// 0 for thin sample(RT), 1 for thick sample(RTT')
int FresnelData::setNewtonParas(int dataNo, int thickFlag) {

    // Debug.
    cout << "void FresnelData::setNewtonParas(int dataNo, int thickFlag)" << endl;

    if(thickFlag == 0) {
        thickness_ = thinThickness_;
        for(int i=0;i<TOTALDATANO;i++) {
            transmittance_[i] = thinTransmittance_[i];
            reflectance_[i] = thinReflectance_[i];
        }
    } else if(thickFlag == 1) {
        thickness_ = thinThickness_;
        for(int i=0;i<TOTALDATANO;i++) {
            transmittance_[i] = thickTransmittance_[i];
            reflectance_[i] = thickReflectance_[i];
        }
    } else {
        cout << "Wrong thick flag!" <<  endl;
        return EXIT_FAILURE;
    }

    for(int i=0;i<6;i++) newtonParas_[i] = 0; // reset/initialize

    double tmpIOR = indexOfRefraction_[dataNo];
    double tmpEC = extinctionCoefficient_[dataNo];
    double atmpFuncValue(0);
    double btmpFuncValue(0);

    newtonParas_[0] = evalCaculatedGrossTransmittance(dataNo) - transmittance_[dataNo];
    newtonParas_[1] = evalCaculatedGrossReflectance(dataNo) - reflectance_[dataNo];

    // T partial n
    indexOfRefraction_[dataNo] += DELTA;
    atmpFuncValue = evalCaculatedGrossTransmittance(dataNo);
    cout << "atmpFuncValue " << atmpFuncValue << endl;
    indexOfRefraction_[dataNo] = tmpIOR;
    indexOfRefraction_[dataNo] -= DELTA;
    btmpFuncValue = evalCaculatedGrossTransmittance(dataNo);
    cout << "btmpFuncValue " << btmpFuncValue << endl;
    indexOfRefraction_[dataNo] = tmpIOR;
    cout << "atmpFuncValue - btmpFuncValue = " << atmpFuncValue - btmpFuncValue << endl;
    newtonParas_[2] = (atmpFuncValue - btmpFuncValue)/ (2*DELTA);

    // T partial k
    extinctionCoefficient_[dataNo] += DELTAEC;
    atmpFuncValue = evalCaculatedGrossTransmittance(dataNo);
    extinctionCoefficient_[dataNo] = tmpEC;
    extinctionCoefficient_[dataNo] -= DELTAEC;
    btmpFuncValue = evalCaculatedGrossTransmittance(dataNo);
    extinctionCoefficient_[dataNo] = tmpEC;
    newtonParas_[3] = (atmpFuncValue - btmpFuncValue)/ (2*DELTAEC);

    // R partial n
    indexOfRefraction_[dataNo] += DELTA;
    atmpFuncValue = evalCaculatedGrossReflectance(dataNo);
    indexOfRefraction_[dataNo] = tmpIOR;
    indexOfRefraction_[dataNo] -= DELTA;
    btmpFuncValue = evalCaculatedGrossReflectance(dataNo);
    indexOfRefraction_[dataNo] = tmpIOR;
    newtonParas_[4] = (atmpFuncValue - btmpFuncValue)/ (2*DELTA);

    // T partial k
    extinctionCoefficient_[dataNo] += DELTAEC;
    atmpFuncValue = evalCaculatedGrossReflectance(dataNo);
    extinctionCoefficient_[dataNo] = tmpEC;
    extinctionCoefficient_[dataNo] -= DELTAEC;
    btmpFuncValue = evalCaculatedGrossReflectance(dataNo);
    extinctionCoefficient_[dataNo] = tmpEC;
    newtonParas_[5] = (atmpFuncValue - btmpFuncValue)/ (2*DELTAEC);

    // Debug.
    for(int i=0;i<6;i++) {
        cout << newtonParas_[i] << endl;
    }

    return NK_SUCCESS;

}

void FresnelData::evalJacobian(int dataNo) {

    double delta = newtonParas_[2]*newtonParas_[5] - newtonParas_[3]*newtonParas_[4];
    if(delta == 0) {
        numericalStatus_[dataNo] = 1;
    } else {
        // Debug. Please check again to avoid keying in wrong
        indexOfRefraction_[dataNo] += (newtonParas_[3]*newtonParas_[1] - newtonParas_[0]*newtonParas_[5])/delta;
        extinctionCoefficient_[dataNo] += (newtonParas_[0]*newtonParas_[4] - newtonParas_[2]*newtonParas_[1])/delta;
        numericalStatus_[dataNo] = 0;
    }

}
