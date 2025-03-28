/****************************************************************************\

    Neumerical solution of index of refraction and extinction coefficient.
    Author  : Taihsiang
    Date    : Apr., 6, 2009

\****************************************************************************/

// Input a wavelength value (nm) to find the numerical solution
#define WHICH_WAVELENGTH 279.0
// Input the thickness of the thiner sample (mm)
#define THIN_THICKNESS 10.113
//#define THIN_THICKNESS 15.150
// Input filename of transmittance of a thin sample
//#define FILENAME_THIN_TRANSMITTANCE "1-1-1-1.csv"
#define FILENAME_THIN_TRANSMITTANCE "gt.dat"
// Input filename of reflectance of a thin sample
//#define FILENAME_THIN_REFLECTANCE "1-1-2-1.csv"
#define FILENAME_THIN_REFLECTANCE "gr.dat"
// Input the thickness of the thicker sample (mm)
#define THICK_THICKNESS 14.80
// Input filename of transmittance of a thick sample
#define FILENAME_THICK_TRANSMITTANCE "2-1-1-1.csv"
// Input filename of reflectance of a thick sample
#define FILENAME_THICK_REFLECTANCE "2-1-2-1.csv"


#include <cmath>
#include <iostream>
#include "FresnelData.h"

using namespace std;

int main(int argc, char *argv[]) {

    FresnelData fresnelData(FILENAME_THIN_TRANSMITTANCE, FILENAME_THIN_REFLECTANCE);
    // Initialize index of refraction, extinction coefficient, and thickness
    //fresnelData.setInitialParas(1.505,0.009*WHICH_WAVELENGTH*1.0e-6/(4.0*M_PI),THIN_THICKNESS, THICK_THICKNESS);
    fresnelData.setInitialParas(1.505,0.009,THIN_THICKNESS);

    cout << "wavelen\tn value\talpha\t\tstatus" << endl;


    for(int i=0;i<TOTALDATANO;i++) {
        //fresnelData.newtonMethodRTRTT(i);
        fresnelData.newtonMethodRT(i);
        fresnelData.dump(i);
    }

    fresnelData.dumpToFile("paras.dat");


    /*
    FresnelData fresnelDataThick(FILENAME_THICK_TRANSMITTANCE, FILENAME_THICK_REFLECTANCE);
    fresnelDataThick.setInitialParas(1.505,0.009,THICK_THICKNESS);

    for(int i=0;i<TOTALDATANO;i++) {
        //fresnelData.newtonMethodRTRTT(i);
        fresnelDataThick.newtonMethodRT(i);
        fresnelDataThick.dump(i);
    }

    fresnelDataThick.dumpToFile("paras2.dat");
    */


    /*
    fresnelData.setSecondInitialParas();

    for(int i=0;i<TOTALDATANO;i++) {
        //fresnelData.newtonMethodRTRTT(i);
        fresnelData.dump(i);
    }

    fresnelData.dumpToFile("paras2.dat");

    for(int i=0;i<TOTALDATANO;i++) {
        fresnelData.newtonMethodRTRTT(i);
        fresnelData.dump(i);
    }

    fresnelData.dumpToFile("paras3.dat");
    */

    // Debug.
    //fresnelData.newtonMethodRTRTTSingleWavelength(WHICH_WAVELENGTH*1.0e-6);
    //fresnelData.dumpSingleWavelengthNK(WHICH_WAVELENGTH*1.0e-6);

    return 0;

}
