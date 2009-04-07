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
// how precise when approaching with Newton method
#define ACCURACY 1.0e-6
// the try-and-error initial IOR and EC range when first initial IOR and EC failed
#define MINIOR 1.4
#define MAXIOR 1.6
#define MINEC 0.0
#define MAXEC 1.0e-6


#define NK_SUCCESS 0
#define NK_ERROR 1

#include <string>

using namespace std;

class FresnelData {
    public:
        FresnelData();
        ~FresnelData();
        FresnelData(string transmittanceFilename, string reflectanceFilename);
        void resetAllPrivate();
        int loadFromFile(string transmittanceFilename, string reflectanceFilename);
        void dump(int dataNo);
        int dumpSingleWavelengthNK(double wavelengthValue);
        void set(double *indexOfRefraction, double *extinctionCoefficient, double thickness);
        void get(double *indexOfRefraction, double *extinctionCoefficient, double thickness);

        double evalInternalTransmittance(int dataNo);
        double evalFrontSurfacePrimaryRelectance(int dataNo);
        double evalCaculatedGrossTransmittance(int dataNo);
        double evalCaculatedGrossReflectance(int dataNo);
        void evalJacobian(int dataNo);
        void setInitialParas(double indexOfRefraction, double extinctionCoefficient, double thinThickness, double thickThickness);
        void setKToRetry(int dataNo, int loopNo);
        void setNKToRetry(int dataNo, int loopNo);
        void newtonMethodRTRTT(int dataNo); // combine RT and RTT' and try different initial parameters till being successful.
        void newtonMethodOneDForK(int dataNo); // RT method
        void newtonMethodTwoDForNK(int dataNo); // RTT' method

    private:
        int effectiveTotalDataNo_; // how many data points were read in

        double thinTransmittance_[TOTALDATANO];
        double thinReflectance_[TOTALDATANO];
        double thickTransmittance_[TOTALDATANO];
        double thickReflectance_[TOTALDATANO];
        double wavelength_[TOTALDATANO];
        double indexOfRefraction_[TOTALDATANO];
        double extinctionCoefficient_[TOTALDATANO];
        double thinThickness_;
        double thickThickness_;

        int numericalStatus_[TOTALDATANO];
        double newtonParas_[6]; // Constraint of T and R, T partial n and k, R partial n and k

};
#endif
