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
        void setInitialParas(double indexOfRefraction, double extinctionCoefficient, double thickness);
        void get(double *indexOfRefraction, double *extinctionCoefficient, double thickness);
        void newtonMethodRTRTT(int dataNo); // combine RT and RTT' and try different initial parameters till being successful.
        void newtonMethodOneDForK(int dataNo); // RT method
        void newtonMethodTwoDForNK(int dataNo); // RTT' method

    private:
        double transmittance_[TOTALDATANO];
        double reflectance_[TOTALDATANO];
        double wavelengthTransmittance_[TOTALDATANO];
        double wavelengthReflectance_[TOTALDATANO];
        double wavelength_[TOTALDATANO];
        double indexOfRefraction_[TOTALDATANO];
        double extinctionCoefficient_[TOTALDATANO];
        double numericalStatus_[TOTALDATANO];
        double thickness_;

};
#endif
