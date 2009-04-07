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

FresnelData::FresnelData(string transmittanceFilename, string reflectanceFilename) {

    resetAllPrivate();
    loadFromFile(transmittanceFilename,reflectanceFilename);

}

void FresnelData::resetAllPrivate() {

    for(int i=0;i<TOTALDATANO;i++) {
        transmittance_[i] = -1;
        reflectance_[i] = -1;
        wavelengthTransmittance_[i] = -1;
        wavelengthReflectance_[i] = -1;
        wavelength_[i] = -1;
        indexOfRefraction_[i] = -1;
        extinctionCoefficient_[i] = -1;
        numericalStatus_[i] = -1;
        thickness_ = -1;
    }
 
}

int FresnelData::loadFromFile(string transmittanceFilename, string reflectanceFilename) {

    cout << "Reading in ...... " << transmittanceFilename << endl;
    cout << "Reading in ...... " << reflectanceFilename << endl;
    ifstream finTransmittance(transmittanceFilename.data());
    ifstream finReflectance(reflectanceFilename.data());

    for(int i=0;i<TOTALDATANO; i++) {
        //cout << "The " << i << "th data" << endl; // Debug.
        finTransmittance >> wavelengthTransmittance_[i];
        finReflectance >> wavelengthReflectance_[i];
        //cout << wavelengthTransmittance_[i] << "\t" << wavelengthReflectance_[i] << endl; // Debug.
        if(wavelengthTransmittance_[i] == wavelengthReflectance_[i]) {
            finTransmittance >> transmittance_[i];
            finReflectance >> reflectance_[i];
            wavelength_[i] = wavelengthTransmittance_[i];
            //cout << transmittance_[i] << "\t" << reflectance_[i] << "\t" << wavelength_[i] << endl; // Debug.
        }
    }

    if (finTransmittance.fail() || finReflectance.fail()) return NK_ERROR;
    finTransmittance.close();
    finReflectance.close();

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

void FresnelData::resetInitialPara(int loopNo) {

    

}

void FresnelData::setInitialParas(double indexOfRefraction, double extinctionCoefficient, double thickness) {

    for(int i=0;i<TOTALDATANO;i++) {
        indexOfRefraction_[i] = indexOfRefraction;
        extinctionCoefficient_[i] = extinctionCoefficient;
    }
    thickness_ = thickness;

}

void FresnelData::get( double *indexOfRefraction, double *extinctionCoefficient, double thickness) {

    for(int i=0;i<TOTALDATANO;i++) {
        indexOfRefraction[i] = indexOfRefraction_[i];
        extinctionCoefficient[i] = extinctionCoefficient_[i];
    }
    thickness = thickness_;

}

void FresnelData::newtonMethodRTRTT(int dataNo) {

    int maxLoop = MAXLOOP;
    do
    {
        newtonMethodOneDForK(dataNo);
        newtonMethodTwoDForNK(dataNo);
        if(numericalStatus_[dataNo]!=0) resetInitialPara(maxLoop);
            
    }
    while(numericalStatus_[dataNo]!=0 && (--maxLoop));


}

void FresnelData::newtonMethodOneDForK(int dataNo) {

    cout << "I am Newton method!" << endl;

}

void FresnelData::newtonMethodTwoDForNK(int dataNo) {



}
