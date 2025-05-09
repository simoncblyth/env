/************************************************************************\

    File: FresnelData.cpp
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
//#define DEBUGMODE

#include <cmath>
#include <cfloat>
#include <fstream>
#include <iostream>
#include <string>
#include "FresnelData.h"

using namespace std;

FresnelData::FresnelData(void) {

}

FresnelData::~FresnelData(void) {

}

FresnelData::FresnelData(   string thinTransmittanceFilename, string thinReflectanceFilename) {

    resetAllPrivate();
    loadFromFile(thinTransmittanceFilename,thinReflectanceFilename);

}

void FresnelData::resetAllPrivate() {

    for(int i=0;i<TOTALDATANO;i++) {
        thinTransmittance_[i] = -1;
        thinReflectance_[i] = -1;
        wavelength_[i] = -1;
        indexOfRefraction_[i] = -1;
        alpha_[i] = -1;
        numericalStatus_[i] = -1;
    }
    thinThickness_ = -1;
}

int FresnelData::loadFromFile(  string thinTransmittanceFilename, string thinReflectanceFilename) {

    cout << "Reading in ...... " << thinTransmittanceFilename << endl;
    cout << "Reading in ...... " << thinReflectanceFilename << endl;
    ifstream finThinTransmittance(thinTransmittanceFilename.data());
    ifstream finThinReflectance(thinReflectanceFilename.data());

    // thinWT:thinWavelengthFromTransmittance
    // thinWR:thinWavelengthFromReflectance
    long double thinWT(0), thinWR(0), thickWT(0), thickWR(0);
    int effectiveTotalDataNo(0);
    for(int i=0;i<TOTALDATANO; i++) {
        finThinTransmittance >> thinWT;
        finThinReflectance >> thinWR;
        if((thinWT == thinWR)) {
            finThinTransmittance >> thinTransmittance_[i] >> thinTransmittanceRMS_[i];
            finThinReflectance >> thinReflectance_[i] >> thinReflectanceRMS_[i];
            // unit %->0.01 nm->mm
            thinTransmittance_[i] = thinTransmittance_[i]*0.01;
            thinReflectance_[i] = thinReflectance_[i]*0.01;
            wavelength_[i] = thinWT*1.0e-6;
            effectiveTotalDataNo++;
        }
    }
    effectiveTotalDataNo_ = effectiveTotalDataNo;
    if(effectiveTotalDataNo_ < 0 || effectiveTotalDataNo_ == 0 ) return EXIT_FAILURE;
    if (finThinTransmittance.fail() || finThinReflectance.fail()) return NK_ERROR;
    finThinTransmittance.close();
    finThinReflectance.close();


    // Debug
    // and try the 8 degrees reflection incident angle deviation
    //for(int i=0;i<0;i++) {
    //    thinReflectance_[i] = thinReflectance_[i] - 0.01;
    //    thickReflectance_[i] = thickReflectance_[i] - 0.01;
    //}

    return NK_SUCCESS;

}

void FresnelData::dump(int dataNo) {

    evalConstrain(dataNo);

    if(numericalStatus_[dataNo] == NK_SUCCESS) {
        cout << 1000000.0*wavelength_[dataNo] << "nm\t" << indexOfRefraction_[dataNo]
        << "\t" << alpha_[dataNo]
        << "\tSUCCESS!!\t"
        << thinTransmittanceConstrain_[dataNo] << "\t" << thinReflectanceConstrain_[dataNo];
        if(transmittance_[dataNo] + reflectance_[dataNo] > 1.0) cout << "\t Y";
        cout << endl;
    } else if(numericalStatus_[dataNo] == NK_ERROR) {
        //cout << 1000000.0*wavelength_[dataNo] << "nm\tN/A\tN/A\tFAILED!!\t"
        cout << 1000000.0*wavelength_[dataNo] << "nm\t" << indexOfRefraction_[dataNo]
        << "\t" << alpha_[dataNo] << "\tFAILED!!\t"
        << thinTransmittanceConstrain_[dataNo] << "\t" << thinReflectanceConstrain_[dataNo];
        if(transmittance_[dataNo] + reflectance_[dataNo] > 1.0) cout << "\t YandFailed!";
        cout << endl;
    } else {
        //cout << 1000000.0*wavelength_[dataNo] << "nm\tN/A\tN/A\tERROR/UNKNOWN\t"
        cout << 1000000.0*wavelength_[dataNo] << "nm\t" << indexOfRefraction_[dataNo]
        << "\t" << alpha_[dataNo] << "\tFAILED/UNKNOEN!!\t"
        << thinTransmittanceConstrain_[dataNo] << "\t" << thinReflectanceConstrain_[dataNo];
        if(transmittance_[dataNo] + reflectance_[dataNo] > 1.0) cout << "\t Y";
        cout << endl;
    }


}

void FresnelData::dumpBasicInfo(int dataNo) {

    cout << 1000000*wavelength_[dataNo] << "nm " << indexOfRefraction_[dataNo] << " " << alpha_[dataNo];

    evalConstrain(dataNo);

    cout << "\t" << thinTransmittanceConstrain_[dataNo] << " " << thinReflectanceConstrain_[dataNo] << endl;

}

void FresnelData::dumpToFile(string outputFilename) {

    ofstream fout;
    fout.open(outputFilename.data());


    for(int dataNo=0;dataNo<TOTALDATANO;dataNo++) {

        if(numericalStatus_[dataNo] == NK_SUCCESS) {
            evalError(dataNo);
        } else {
            indexOfRefractionErrorUp_[dataNo] = NK_ERROR;
            alphaErrorUp_[dataNo] = NK_ERROR;
            indexOfRefractionErrorLow_[dataNo] = NK_ERROR;
            alphaErrorLow_[dataNo] = NK_ERROR;
        }

        long double att, attUp, attLow;
        att = 1.0/alpha_[dataNo];
        attLow = 1.0/(alpha_[dataNo] + alphaErrorUp_[dataNo]);
        if((alpha_[dataNo] - alphaErrorUp_[dataNo]) > 0) {
            attUp = 1.0/(alpha_[dataNo] - alphaErrorUp_[dataNo]);
        } else { 
            attUp = att;
            //numericalStatus_[dataNo] = NK_ERROR; // reject solutions which error domains too much
        }

        fout << 1000000.0*wavelength_[dataNo]
        << " " << indexOfRefraction_[dataNo]
        << " " << indexOfRefractionErrorUp_[dataNo]
        //<< " " << indexOfRefractionErrorLow_[dataNo]
        << " " << alpha_[dataNo]
        << " " << alphaErrorUp_[dataNo]
        << " " << att
        << " " << attUp - att
        << " " << att - attLow
        //<< " " << alphaErrorLow_[dataNo]
        << " " << thinTransmittanceConstrain_[dataNo] 
        << " " << thinReflectanceConstrain_[dataNo]
        << " " << numericalStatus_[dataNo]
        << endl;
    }


}


void FresnelData::setNKToRetry(int dataNo, int loopNo) {

    //cout << "void FresnelData::setNKToRetry(int dataNo, int loopNo)" << endl;
    long double rangeIOR = MAXIOR - MINIOR;
    long double rangeEC = MAXEC - MINEC;
    long double ratio = (long double)(loopNo)/(long double)(MAXLOOP);
    indexOfRefraction_[dataNo] = MINIOR + ratio*rangeIOR;
    alpha_[dataNo] = MINEC + ratio*rangeEC;

}

void FresnelData::setInitialParas(  long double indexOfRefraction, long double alpha, long double thinThickness) {

    for(int i=0;i<TOTALDATANO;i++) {
        indexOfRefraction_[i] = indexOfRefraction;
        alpha_[i] = alpha;
    }
    thinThickness_ = thinThickness;
}


long double FresnelData::evalInternalTransmittance(int dataNo) {

    long double extinctionCoefficient = AlphaTok(dataNo);
    long double internalTransmittance(-1);
    internalTransmittance = exp((-4.0*M_PI*extinctionCoefficient*thickness_)/wavelength_[dataNo]);

    return internalTransmittance;

}

long double FresnelData::evalFrontSurfacePrimaryRelectance(int dataNo) {

    long double extinctionCoefficient = AlphaTok(dataNo);

    long double frontSurfacePrimaryRelectance(-1);
    long double numerator(0);
    long double denominator(0);
    numerator = (indexOfRefraction_[dataNo] - 1.0)*(indexOfRefraction_[dataNo] - 1.0)
                + extinctionCoefficient*extinctionCoefficient;
    denominator =   (indexOfRefraction_[dataNo] + 1.0)*(indexOfRefraction_[dataNo] + 1.0)
                    + extinctionCoefficient*extinctionCoefficient;
    frontSurfacePrimaryRelectance = numerator/denominator;


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

void FresnelData::newtonMethodRT(int dataNo) {

    int maxLoop = MAXLOOP;
    do
    {
        if(maxLoop != MAXLOOP) setNKToRetry(dataNo, maxLoop);
        newtonMethodTwoDForNK(dataNo, MAXLOOP - maxLoop);
        setCandidatePerEvent(dataNo, MAXLOOP - maxLoop);
    }
    while((--maxLoop));

    setCandidate(dataNo, MAXLOOP);

    //if((validateFinalValue(dataNo) == NK_SUCCESS)) {
    //    numericalStatus_[dataNo] = NK_SUCCESS;
    //} else {
    //    numericalStatus_[dataNo] = NK_ERROR;
    //}

}


void FresnelData::newtonMethodTwoDForNK(int dataNo, int loop) {

    int maxLoop = MAXLOOP;
    long double delta(0), tmpDeltaIOR(0), tmpDeltaEC(0);
    do
    {
        if(setNewtonParas(dataNo,0,2) == NK_SUCCESS) {
            evalJacobian(dataNo);
            delta = newtonParas_[2]*newtonParas_[5] - newtonParas_[3]*newtonParas_[4];
            if(delta == 0) {
                maxLoop = NK_ERROR; // forcing to stop
            }
        } else {
            maxLoop = NK_ERROR;
        }
    }
    while(((fabs(newtonParas_[6]) > NACCURACY) || (fabs(newtonParas_[7]) > KACCURACY)) && (--maxLoop));

    if(maxLoop > 0) {
        numericalStatus_[dataNo] = NK_SUCCESS;
        loopStatus_[loop] = NK_SUCCESS;
    } else {
        numericalStatus_[dataNo] = NK_ERROR;
        loopStatus_[loop] = NK_ERROR;
    }

}

int FresnelData::setNewtonParas(int dataNo, int thickFlag, int methodDimensionFlag) {


    if(methodDimensionFlag != 1 && methodDimensionFlag != 2) {
        cout << "Wrong methodDimensionFlag" << endl;
        return EXIT_FAILURE;
    }

    thickness_ = thinThickness_;
    for(int i=0;i<TOTALDATANO;i++) {
        transmittance_[i] = thinTransmittance_[i];
        reflectance_[i] = thinReflectance_[i];
    }

    for(int i=0;i<6;i++) newtonParas_[i] = 0; // reset/initialize

    long double tmpIOR = indexOfRefraction_[dataNo];
    long double tmpEC = alpha_[dataNo];
    long double atmpFuncValue(0);
    long double btmpFuncValue(0);



    if(validateNumericalRange(dataNo) == NK_SUCCESS) { // avoid nan and inf


        newtonParas_[0] = evalTransmittanceConstrain(dataNo);
        newtonParas_[1] = evalReflectanceConstrain(dataNo);
 
        // T partial k
        alpha_[dataNo] += DELTAEC;
        atmpFuncValue = evalTransmittanceConstrain(dataNo);
        alpha_[dataNo] = tmpEC;
        alpha_[dataNo] -= DELTAEC;
        btmpFuncValue = evalTransmittanceConstrain(dataNo);
        alpha_[dataNo] = tmpEC;
        newtonParas_[3] = (atmpFuncValue - btmpFuncValue)/ (2*DELTAEC);
     
        if(methodDimensionFlag == 2) {
        // T partial n
        indexOfRefraction_[dataNo] += DELTA;
        atmpFuncValue = evalTransmittanceConstrain(dataNo);
        indexOfRefraction_[dataNo] = tmpIOR;
        indexOfRefraction_[dataNo] -= DELTA;
        btmpFuncValue = evalTransmittanceConstrain(dataNo);
        indexOfRefraction_[dataNo] = tmpIOR;
        newtonParas_[2] = (atmpFuncValue - btmpFuncValue)/ (2*DELTA);

        // R partial n
        indexOfRefraction_[dataNo] += DELTA;
        atmpFuncValue = evalReflectanceConstrain(dataNo);
        indexOfRefraction_[dataNo] = tmpIOR;
        indexOfRefraction_[dataNo] -= DELTA;
        btmpFuncValue = evalReflectanceConstrain(dataNo);
        indexOfRefraction_[dataNo] = tmpIOR;
        newtonParas_[4] = (atmpFuncValue - btmpFuncValue)/ (2*DELTA);
     
        // R partial k
        alpha_[dataNo] += DELTAEC;
        atmpFuncValue = evalReflectanceConstrain(dataNo);
        alpha_[dataNo] = tmpEC;
        alpha_[dataNo] -= DELTAEC;
        btmpFuncValue = evalReflectanceConstrain(dataNo);
        alpha_[dataNo] = tmpEC;
        newtonParas_[5] = (atmpFuncValue - btmpFuncValue)/ (2*DELTAEC);
        }
        return NK_SUCCESS;
    } else {
        numericalStatus_[dataNo] = NK_ERROR;
        return NK_ERROR;
    }


}

void FresnelData::setCandidatePerEvent(int dataNo, int loop) {

    candidatesPerEvent_[loop][0] = indexOfRefraction_[dataNo];
    candidatesPerEvent_[loop][1] = alpha_[dataNo];

}

void FresnelData::setCandidate(int dataNo, int loopEff) {

    long double chiConstrainTmp(0.0);
    int bestCandidateLoopNo(0);

    chiConstrain_ = 100.0;

    int numericalCheker(0);

    for(int i=0;i<loopEff;i++) {
        indexOfRefraction_[dataNo] = candidatesPerEvent_[i][0];
        alpha_[dataNo] = candidatesPerEvent_[i][1];
        //if(dataNo == 19) cout << " " << alpha_[dataNo] << " " << loopStatus_[i] << endl;
        if(validateFinalValue(dataNo) == NK_SUCCESS && loopStatus_[i] == NK_SUCCESS) {
            //if(dataNo == 19)  cout << "over here " << endl;
            evalConstrain(dataNo);
            chiConstrainTmp = (fabs(thinTransmittanceConstrain_[dataNo]) + fabs(thinReflectanceConstrain_[dataNo]));
            if(chiConstrain_ > chiConstrainTmp) {
                chiConstrain_ = chiConstrainTmp;
                bestCandidateLoopNo = i;

                // alpha may change sigh
                candidatesPerEvent_[i][0] = indexOfRefraction_[dataNo];
                candidatesPerEvent_[i][1] = alpha_[dataNo];
                numericalCheker++;
            }
        }
    }

    // if no one converge
    if(numericalCheker == 0) { 
        numericalStatus_[dataNo] = NK_ERROR;

    } else { numericalStatus_[dataNo] = NK_SUCCESS;}

    indexOfRefraction_[dataNo] = candidatesPerEvent_[bestCandidateLoopNo][0];
    alpha_[dataNo] = candidatesPerEvent_[bestCandidateLoopNo][1];

}

// some k/alpha is negative but really close to 0 (small enough to be reasonable), try...
int FresnelData::retryNegativeK(int dataNo) {

    long double alphatmp = alpha_[dataNo];
    alpha_[dataNo] = -alpha_[dataNo];

    if(evalConstrain(dataNo) == NK_SUCCESS) {    
        return NK_SUCCESS;
    } else { 
        alpha_[dataNo] = alphatmp;
        return NK_ERROR;
    }


}

// 1.validate value physically legal or not
//      n shold be 1~2, alpha should be positive and attenuation length DONT be larger than 20m
// 2.final decision to accept the result of not
int FresnelData::validateFinalValue(int dataNo) {

    // some k/alpha is negative but really close to 0 (small enough to be reasonable), try...
    if((alpha_[dataNo] < 0.0)) {
        //cout << "alpha is negative...." << endl;
        if(retryNegativeK(dataNo) == NK_SUCCESS) {
            if(validateValue(dataNo) == NK_SUCCESS) { return NK_SUCCESS;} else { return NK_ERROR;}
        } else {
            return NK_ERROR;
        }
    } else {
        if(validateValue(dataNo) == NK_SUCCESS) { return NK_SUCCESS;} else { return NK_ERROR;}
    }

}

int FresnelData::validateValue(int dataNo) {

    // different wavelength should has different validation...phsics
    if(dataNo < CUTOFF) {
        if((indexOfRefraction_[dataNo] > N_MIN) && (indexOfRefraction_[dataNo] < N_MAX) && (alpha_[dataNo] > ALPHA_RED_MIN)) {
            return NK_SUCCESS;
        } else {
            //cout << "Beyond physics requirement <CUTOFF" << endl;
            return NK_ERROR;
        }
    } else {
        if((indexOfRefraction_[dataNo] > N_MIN) && (indexOfRefraction_[dataNo] < N_MAX) && (alpha_[dataNo] > ALPHA_BLUE_MIN)) {
            return NK_SUCCESS;
        } else {
            //cout << "Beyond physics requirement >CUTOFF" << endl;
            return NK_ERROR;
        }
    }


}

// validate value not to be nan or inf
// 0 for o.k. 1 for invalid
int FresnelData::validateNumericalRange(int dataNo) {

    //cout << "Validation of parameters ";
    //if(dataNo < 30) cout << "n is " << indexOfRefraction_[dataNo] << " ,k is " << alpha_[dataNo] << endl;
    if(fabs(indexOfRefraction_[dataNo]) > NEWTON_N || fabs(alpha_[dataNo]) > NEWTON_ALPHA) {
        //cout << "Values are invalid! " << endl;
        return NK_ERROR;
    } else {
        //cout << "values are valid " << endl;
        return NK_SUCCESS;
    }

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
        alpha_[dataNo] += newtonParas_[7];
        //cout << "alpha_[dataNo] " << alpha_[dataNo] << endl;
        numericalStatus_[dataNo] = NK_SUCCESS;
    }

}

int FresnelData::evalConstrain(int dataNo) {

    long double thicknessTmp = thickness_;

    thickness_ = thinThickness_;
    thinTransmittanceConstrain_[dataNo] = evalTransmittanceConstrain(dataNo);
    thinReflectanceConstrain_[dataNo] = evalReflectanceConstrain(dataNo);

    thickness_ = thicknessTmp;

    // Debug
    //cout << "evalConstrain(int dataNo)\t" << thinTransmittanceConstrain_[dataNo] << " " << thinReflectanceConstrain_[dataNo];
    //cout << "\t" << thickTransmittanceConstrain_[dataNo] << " " << thickReflectanceConstrain_[dataNo] << endl;

    if((fabs(thinTransmittanceConstrain_[dataNo]) < TACCURACY) && (fabs(thinReflectanceConstrain_[dataNo]) < RACCURACY)) {
        //cout << endl << "fabs " << thinReflectanceConstrain_[dataNo]  << " fabs" << endl;
        //cout << "evalConstrain(int dataNo): true" << endl;
        return NK_SUCCESS;
    } else {
        return NK_ERROR;
    }
}


long double FresnelData::evalTransmittanceConstrain(int dataNo) {

    return evalCaculatedGrossTransmittance(dataNo) - transmittance_[dataNo];

}

long double FresnelData::evalReflectanceConstrain(int dataNo) {

    return evalCaculatedGrossReflectance(dataNo) - reflectance_[dataNo];

}

long double FresnelData::AlphaTok(int dataNo) {

    return (alpha_[dataNo]*wavelength_[dataNo])/(4.0*M_PI);

}

void FresnelData::evalError(int dataNo) {

    long double tmpIOR = indexOfRefraction_[dataNo];
    long double tmpEC = alpha_[dataNo];
    long double atmpFuncValue(0);
    long double btmpFuncValue(0);

    long double tk(0), tn(0), rk(0), rn(0);


    // T partial k
    alpha_[dataNo] += DELTAEC;
    atmpFuncValue = evalCaculatedGrossTransmittance(dataNo);
    alpha_[dataNo] = tmpEC;
    alpha_[dataNo] -= DELTAEC;
    btmpFuncValue = evalCaculatedGrossTransmittance(dataNo);
    alpha_[dataNo] = tmpEC;
    tk = (atmpFuncValue - btmpFuncValue)/ (2*DELTAEC);

    // T partial n
    indexOfRefraction_[dataNo] += DELTA;
    atmpFuncValue = evalCaculatedGrossTransmittance(dataNo);
    indexOfRefraction_[dataNo] = tmpIOR;
    indexOfRefraction_[dataNo] -= DELTA;
    btmpFuncValue = evalCaculatedGrossTransmittance(dataNo);
    indexOfRefraction_[dataNo] = tmpIOR;
    tn = (atmpFuncValue - btmpFuncValue)/ (2*DELTA);

    // R partial k
    alpha_[dataNo] += DELTAEC;
    atmpFuncValue = evalCaculatedGrossReflectance(dataNo);
    alpha_[dataNo] = tmpEC;
    alpha_[dataNo] -= DELTAEC;
    btmpFuncValue = evalCaculatedGrossReflectance(dataNo);
    alpha_[dataNo] = tmpEC;
    rk = (atmpFuncValue - btmpFuncValue)/ (2*DELTAEC);

    // R partial n
    indexOfRefraction_[dataNo] += DELTA;
    atmpFuncValue = evalCaculatedGrossReflectance(dataNo);
    indexOfRefraction_[dataNo] = tmpIOR;
    indexOfRefraction_[dataNo] -= DELTA;
    btmpFuncValue = evalCaculatedGrossReflectance(dataNo);
    indexOfRefraction_[dataNo] = tmpIOR;
    rn = (atmpFuncValue - btmpFuncValue)/ (2*DELTA);

    // find extreme in Cramer's rule
    if((TERROR*rk)*(RERROR*tk) > 0) {
        if(fabs(TERROR*rk) > fabs(RERROR*tk)) {
            indexOfRefractionErrorUp_[dataNo] = (TERROR*rk)/(tn*rk-tk*rn);
        } else { indexOfRefractionErrorUp_[dataNo] = (-tk*RERROR)/(tn*rk-tk*rn);}
    } else { indexOfRefractionErrorUp_[dataNo] = (TERROR*rk-tk*RERROR)/(tn*rk-tk*rn);}

    if((tn*RERROR)*(TERROR*rn) > 0) {
        if(fabs(RERROR*tn) > fabs(TERROR*rn)) {
            alphaErrorUp_[dataNo] = (tn*RERROR)/(tn*rk-tk*rn);
        } else { alphaErrorUp_[dataNo] = (-TERROR*rn)/(tn*rk-tk*rn);}
    } else { alphaErrorUp_[dataNo] = (tn*RERROR-TERROR*rn)/(tn*rk-tk*rn);}

    indexOfRefractionErrorUp_[dataNo] = fabs(indexOfRefractionErrorUp_[dataNo]);
    alphaErrorUp_[dataNo] = fabs(alphaErrorUp_[dataNo]);

}
