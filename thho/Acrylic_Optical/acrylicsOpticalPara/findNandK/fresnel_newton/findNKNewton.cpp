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
    double thinWT(0), thinWR(0), thickWT(0), thickWR(0), effectiveTotalDataNo(0);
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
    for(int i=0;i<TOTALDATANO;i++) {
        if(wavelength_[i] == wavelengthValue) dataNo = i;
    }

    if(dataNo < -1) {
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

    return 0;

}

void FresnelData::set(double *indexOfRefraction, double *extinctionCoefficient, double thickness) {

    for(int i=0;i<TOTALDATANO;i++) {
        indexOfRefraction_[i] = indexOfRefraction[i];
        extinctionCoefficient_[i] = extinctionCoefficient[i];
    }

}

void FresnelData::setKToRetry(int dataNo, int loopNo) {

    double range = MAXEC - MINEC;
    double ratio = double(loopNO)/double(MAXLOOP);
    extinctionCoefficient_[dataNo] = MINEC + range*ratio;

}

void FresnelData::setNKToRetry(int dataNo, int loopNo) {

    double rangeIOR = MAXIOR - MINIOR;
    double rangeEC = MAXEC - MINEC;
    double ratio = double(loopNO)/double(MAXLOOP);
    indexOfRefraction_[i] = MINIOR + ratio*rangeIOR;
    extinctionCoefficient_ = MINEC + ratio*rangeEC;

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

void FresnelData::get( double *indexOfRefraction, double *extinctionCoefficient, double thickness) {

    for(int i=0;i<TOTALDATANO;i++) {
        indexOfRefraction[i] = indexOfRefraction_[i];
        extinctionCoefficient[i] = extinctionCoefficient_[i];
    }
    thickness = thickness_;

}

double FresnelData::evalInternalTransmittance(int dataNo) {

    double internalTransmittance(-1);
    internalTransmittance = exp((-4.0*M_PI*extinctionCoefficient_[dataNo])/wavelength_[dataNo]);
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
    return caculatedGrossreflectance;

}

void FresnelData::newtonMethodRTRTT(int dataNo) {

    int maxLoop = MAXLOOP;
    do
    {
        newtonMethodOneDForK(dataNo);
        newtonMethodTwoDForNK(dataNo);
        // retry RTT' first
        while(numericalStatus_[dataNo]!=0 && (--maxLoop))
        {
            setKToRetry(maxLoop);
            newtonMethodTwoDForNK(dataNo);
        }
        // if retrying RTT' failed, using the slower but larger range scanning
        while(numericalStatus_[dataNo]!=0 && (--maxLoop))
        {
            setNKToRetry(maxLoop);
            newtonMethodOneDForK(dataNo);
            newtonMethodTwoDForNK(dataNo);
        }   
    }
    while(numericalStatus_[dataNo]!=0 && (--maxLoop));


}

void FresnelData::newtonMethodOneDForK(int dataNo) {

    int maxLoop = MAXLOOP;
    do
    {
        setNewtonParas(dataNo);
        if(newtonParas_[3] != 0) {
            extinctionCoefficient_[dataNo] += newtonParas_[0]/newtonParas_[3];
        } else {
            maxLoop = 0; // forcing to stop
            numericalStatus_[dataNo] = 1;
        }
    }
    while(fabs(newtonParas_[0]/newtonParas_[3]) > ACCURACY && --maxLoop);

    if(maxLoop > 0) {
        numericalStatus_[dataNo] = 0;
    } else {
        numericalStatus_[dataNo] = 1;
    }

}

void FresnelData::newtonMethodTwoDForNK(int dataNo) {

    int maxLoop = MAXLOOP;
    double delta(0), tmpDeltaIOR(0), tmpDeltaEC(0);
    do
    {
        setNewtonParas(dataNo);
        evalJacobian(dataNo);
        delta = newtonParas_[2]*newtonParas_[5] - newtonParas_[3]*newtonParas_[4];
        if(delta == 0) maxLoop = 0; // forcing to stop
        tmpDeltaIOR = (newtonParas_[3]*newtonParas_[1] - newtonParas_[2]*newtonParas_[1])/delta;
        tmpDeltaEC = (newtonParas_[0]*newtonParas_[4] - newtonParas_[0]*newtonParas_[5])/delta;
    }
    while((fabs(tmpDeltaIOR) > ACCURACY) && (fabs(tmpDeltaEC)> ACCURACY) && (--maxLoop))

    if(maxLoop > 0) {
        numericalStatus_[dataNo] = 0;
    } else {
        numericalStatus_[dataNo] = 1;
    }

}

void FresnelData::setNewtonParas(int dataNo) {

    for(int i=0;i<6;i++) newtonParas_[i] = 0; // reset/initialize

    double tmpIOR = indexOfRefraction_[dataNo];
    double tmpEC = extinctionCoefficient_[dataNo];
    double atmpFuncValue(0);
    double btmpFuncValue(0);

    newtonParas_[0] = evalCaculatedGrossTransmittance(dataNo);
    newtonParas_[1] = evalCaculatedGrossReflectance(dataNo);

    // T partial n
    indexOfRefraction_[dataNo] += DBL_MIN;
    atmpFuncValue = evalCaculatedGrossTransmittance(dataNo);
    indexOfRefraction_[dataNo] = tmpIOR;
    indexOfRefraction_[dataNo] -= DBL_MIN;
    btmpFuncValue = evalCaculatedGrossTransmittance(dataNo);
    indexOfRefraction_[dataNo] = tmpIOR;
    newtonParas_[2] = (atmpFuncValue - btmpFuncValue)/ DBL_MIN;

    // T partial k
    extinctionCoefficient_[dataNo] += DBL_MIN;
    atmpFuncValue = evalCaculatedGrossTransmittance(dataNo);
    extinctionCoefficient_[dataNo] = tmpEC;
    extinctionCoefficient_[dataNo] -= DBL_MIN;
    btmpFuncValue = evalCaculatedGrossTransmittance(dataNo);
    extinctionCoefficient_[dataNo] = tmpEC;
    newtonParas_[3] = (atmpFuncValue - btmpFuncValue)/ DBL_MIN;

    // R partial n
    indexOfRefraction_[dataNo] += DBL_MIN;
    atmpFuncValue = evalCaculatedGrossReflectance(dataNo);
    indexOfRefraction_[dataNo] = tmpIOR;
    indexOfRefraction_[dataNo] -= DBL_MIN;
    btmpFuncValue = evalCaculatedGrossReflectance(dataNo);
    indexOfRefraction_[dataNo] = tmpIOR;
    newtonParas_[4] = (atmpFuncValue - btmpFuncValue)/ DBL_MIN;

    // T partial k
    extinctionCoefficient_[dataNo] += DBL_MIN;
    atmpFuncValue = evalCaculatedGrossReflectance(dataNo);
    extinctionCoefficient_[dataNo] = tmpEC;
    extinctionCoefficient_[dataNo] -= DBL_MIN;
    btmpFuncValue = evalCaculatedGrossReflectance(dataNo);
    extinctionCoefficient_[dataNo] = tmpEC;
    newtonParas_[5] = (atmpFuncValue - btmpFuncValue)/ DBL_MIN;

}

void FresnelData::evalJacobian(int dataNo) {

    double delta = newtonParas_[2]*newtonParas_[5] - newtonParas_[3]*newtonParas_[4];
    if(delta == 0) {
        numericalStatus_[dataNo] = 1;
    } else {
        // Debug. Please check again to avoid keying in wrong
        indexOfRefraction_[dataNo] += (newtonParas_[3]*newtonParas_[1] - newtonParas_[2]*newtonParas_[1])/delta;
        extinctionCoefficient_[dataNo] += (newtonParas_[0]*newtonParas_[4] - newtonParas_[0]*newtonParas_[5])/delta;
        numericalStatus_[dataNo] = 0;
    }

}
