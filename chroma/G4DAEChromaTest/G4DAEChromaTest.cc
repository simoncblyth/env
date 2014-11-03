#include <stdio.h>  
#include <stdlib.h>    

#include "Chroma/ChromaPhotonList.hh"
#include "G4DAEChroma/G4DAEPhotonList.hh"
#include "G4ThreeVector.hh"

#include <sstream>
#include <iostream>
#include <vector>

using namespace std ;


void loadphotons(const char* evtkey)
{
   ChromaPhotonList* cpl = ChromaPhotonList::Load(evtkey);
   if(!cpl){
       printf("failed to load photons with evtkey %s \n", evtkey);
   }
   cpl->Print();
}


void gpl_save()
{
    G4DAEPhotonList* gpl = new G4DAEPhotonList(5);

    G4ThreeVector pos(3,3,3);
    G4ThreeVector dir(0,0,1);
    G4ThreeVector pol(0,0,1);

    float time = 1.; 
    float wavelength = 550.; 
    int pmtid = 0x1010101 ;

    gpl->AddPhoton( pos, dir, pol, time, wavelength, pmtid );
    gpl->AddPhoton( pos, dir, pol, time, wavelength, pmtid );
    gpl->AddPhoton( pos, dir, pol, time, wavelength, pmtid );
    gpl->AddPhoton( pos, dir, pol, time, wavelength, pmtid );
    gpl->AddPhoton( pos, dir, pol, time, wavelength, pmtid );

    gpl->Save("gdct001");

    delete gpl ;  

/*
    python -c "import numpy as np ; np.set_printoptions(precision=3, suppress=True) ; print np.load('/usr/local/env/tmp/gdct001.npy')"
    python -c "import numpy as np ; np.set_printoptions(precision=3, suppress=True) ; a = np.load('/usr/local/env/tmp/gdct001.npy') ; print a ; print a.view(np.int32)"
*/

}




int main(int argc, char** argv)
{
    G4DAEPhotonList* gpl = G4DAEPhotonList::Load("gdct001");
    cout << "gpl " << (void*)gpl << endl ;


    return 0 ; 
}

