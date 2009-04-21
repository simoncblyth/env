/************************************************************************\

    File: findNKNewton.h
    Created by: Taihsiang
    Date: Apr., 06, 2009
    Description:
                C++ header to use Fresnel equation
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
            IOR,EC,dataNo(data number)

\************************************************************************/

#ifndef N_K_FRESNEL_NEWTON_H
#define N_K_FRESNEL_NEWTON_H

// how many data points are there? 200nm ~ 800nm, 1 data point/nm, so 601
#define TOTALDATANO 601
// how many loops when iterating Newton / RT / RTT' method
#define MAXLOOP 100
// how precise caculated T and R is when approaching with Newton method, in order to make
// sure a solution is meaningful
// because of the uncertainty of different instrucment
// between R and T, there should be different tolerance..
// tolerance of transmittance
#define TACCURACY 1.0e-2
// tolerance of reflectance
#define RACCURACY 1.0e-3
// how precise n is when approaching with Newton method, in order to make sure a local min/max
#define NACCURACY 1.0e-4
// how precise alpha is when approaching with Newton method, in order to make sure a local min/max
// Please note it now means how precise alpha is.. not k
#define KACCURACY 1.0e-4
// small amount to access differentiation
#define DELTA 1.0e-9
#define DELTAEC 1.0e-14
// presion to compare numbers
#define NUMBER_EQUAL 1.0e-10
// the famous cut-off edge starting at which wavelength...dataNo as indicating index
// 800 - wavelength(nm) = dataNo
// so it should be integer
#define CUTOFF 450
// the try-and-error initial IOR and EC range when first initial IOR and EC failed
#define MINIOR 1.4
#define MAXIOR 1.6
#define MINEC 1.0e-5
#define MAXEC 1.0e-1


#define NK_SUCCESS 0
#define NK_ERROR 1

#include <string>

using namespace std;

class FresnelData {
    public:
        FresnelData();
        ~FresnelData();
        FresnelData(string thinTransmittanceFilename, string thinReflectanceFilename,
                    string thickTransmittanceFilename, string thickReflectanceFilename);
        void resetAllPrivate();
        int loadFromFile(   string thinTransmittanceFilename, string thinReflectanceFilename,
                            string thickTransmittanceFilename, string thickReflectanceFilename);
        void dump(int dataNo);
        void dumpBasicInfo(int dataNo);
        void dumpToFile(string outputFilename);
        int dumpSingleWavelengthNK(long double wavelengthValue);
        void set(long double *indexOfRefraction, long double *extinctionCoefficient);
        void get(long double *indexOfRefraction, long double *extinctionCoefficient);

        long double AlphaTok(int dataNo);
        long double evalInternalTransmittance(int dataNo);
        long double evalFrontSurfacePrimaryRelectance(int dataNo);
        long double evalCaculatedGrossTransmittance(int dataNo);
        long double evalCaculatedGrossReflectance(int dataNo);
        int evalConstrain(int dataNo);
        long double evalTransmittanceConstrain(int dataNo);
        long double evalReflectanceConstrain(int dataNo);
        void evalJacobian(int dataNo);

        void setInitialParas(long double indexOfRefraction, long double extinctionCoefficient, long double thinThickness, long double thickThickness);
        void setAlphaToEC(int dataNo);
        int setSecondInitialParas(void);
        void setKToRetry(int dataNo, int loopNo);
        void setNKToRetry(int dataNo, int loopNo);
        int setNewtonParas(int dataNo, int thickFlag, int methodDimensionFlag);

        int retryNegativeK(int dataNo);

        void newtonMethodRT(int dataNo);
        void newtonMethodRTRTT(int dataNo); // combine RT and RTT' and try different initial parameters till being successful.
        int newtonMethodRTRTTSingleWavelength(long double wavelengthValue);
        void newtonMethodOneDForK(int dataNo); // RT method
        void newtonMethodTwoDForNK(int dataNo); // RTT' method

        int validateFinalValue(int dataNo);
        int validateValue(int dataNo);
        int validateNumericalRange(int dataNo);

        // Debug.
        // Unit Test
        // 1D unit test
        void setNewtonParasForOneDUnitTest(int dataNo);
        long double evalOneDUnitTestFormula(int dataNo);
        void dumpOneDUnitTest(int dataNo);
        // 2D unit test
        void setNewtonParasForTwoDUnitTest(int dataNo);
        long double evalTwoDUnitTestFormulaOne(int dataNo);
        long double evalTwoDUnitTestFormulaTwo(int dataNo);
        void dumpTwoDUnitTest(int dataNo);


    private:
        int effectiveTotalDataNo_; // how many data points were read in

        long double thinTransmittance_[TOTALDATANO];
        long double thinReflectance_[TOTALDATANO];
        long double thickTransmittance_[TOTALDATANO];
        long double thickReflectance_[TOTALDATANO];
        long double wavelength_[TOTALDATANO]; // mm
        long double indexOfRefraction_[TOTALDATANO];
        //long double extinctionCoefficient_[TOTALDATANO];
        long double alpha_[TOTALDATANO]; // 1/mm
        long double transmittance_[TOTALDATANO];
        long double reflectance_[TOTALDATANO];
        long double thickness_; // mm
        long double thinThickness_; // mm
        long double thickThickness_; // mm
        long double thinThickFlag_; // 0 for thin sample and 1 for thick sample

        long double thinTransmittanceConstrain_[TOTALDATANO];
        long double thickTransmittanceConstrain_[TOTALDATANO];
        long double thinReflectanceConstrain_[TOTALDATANO];
        long double thickReflectanceConstrain_[TOTALDATANO];
        int numericalStatus_[TOTALDATANO];
        long double newtonParas_[8]; // Constraint of T and R, T partial n and k, R partial n and k, dn and dk

};
#endif
