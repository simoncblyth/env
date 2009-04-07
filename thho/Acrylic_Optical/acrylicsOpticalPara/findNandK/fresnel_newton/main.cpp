/****************************************************************************\

    Neumerical solution of index of refraction and extinction coefficient.
    Author  : Taihsiang
    Date    : Apr., 6, 2009

\****************************************************************************/

// Input a wavelength value (nm) to find the numerical solution
#define WHICH_WAVELENGTH 798
// Input the thickness of the thiner sample
#define THIN_THICKNESS 10.0
// Input filename of transmittance of a thin sample
#define FILENAME_THIN_TRANSMITTANCE "1-1-1-1.csv"
// Input filename of reflectance of a thin sample
#define FILENAME_THIN_REFLECTANCE "1-1-2-1.csv"
// Input the thickness of the thicker sample
#define THICK_THICKNESS 15.0
// Input filename of transmittance of a thick sample
#define FILENAME_THICK_TRANSMITTANCE "2-1-1-1.csv"
// Input filename of reflectance of a thick sample
#define FILENAME_THICK_REFLECTANCE "2-1-2-1.csv"



#include "findNKNewton.h"

using namespace std;

int main(int argc, char *argv[]) {

    FresnelData fresnelData(FILENAME_THIN_TRANSMITTANCE, FILENAME_THIN_REFLECTANCE);
    // Initialize index of refraction, extinction coefficient, and thickness
    fresnelData.setInitialParas(1.5,1.0e-6,THIN_THICKNESS);

    /*
    for(int i=0;i<TOTALDATANO;i++) {
        fresnelData.newtonMethod(i);
        fresnelData.dump(i);
    }
    */

    // Debug.
    fresnelData.newtonMethod(WHICH_WAVELENGTH);
    fresnelData.dumpSingleWavelengthNK(WHICH_WAVELENGTH);

    return 0;

}
