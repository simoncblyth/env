#include <stdio.h>  
#include <stdlib.h>    

//#include "Chroma/ChromaPhotonList.hh"
#include "G4DAEChroma/G4DAEPhotonList.hh"
#include "G4DAEChroma/G4DAESocket.hh"
#include "G4DAEChroma/G4DAECommon.hh"
#include "G4ThreeVector.hh"

#include <sstream>
#include <iostream>
#include <vector>

using namespace std ;

/*
void loadphotons(const char* evtkey)
{
   ChromaPhotonList* cpl = ChromaPhotonList::Load(evtkey);
   if(!cpl){
       printf("failed to load photons with evtkey %s \n", evtkey);
   }
   cpl->Print();
}
*/


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

    gpl->Save("gdct002");

    delete gpl ;  

/*
    python -c "import numpy as np ; np.set_printoptions(precision=3, suppress=True) ; print np.load('/usr/local/env/tmp/gdct001.npy')"
    python -c "import numpy as np ; np.set_printoptions(precision=3, suppress=True) ; a = np.load('/usr/local/env/tmp/gdct001.npy') ; print a ; print a.view(np.int32)"
*/

}


int gpl_load()
{
    G4DAEPhotonList* gpl = G4DAEPhotonList::Load("gdct001");
    if(!gpl) return 1 ; 

    gpl->Print();
    gpl->Details(0);
    gpl->DumpBuffer();

    return 0 ; 
}


int gpl_network()
{
    const char* frontend = "FRONTEND" ; 
    const char* backend  = "BACKEND" ;

    bool is_frontend = getenv(frontend) ;
    bool is_backend = getenv(backend) ;

    cout << "is_frontend " << is_frontend << endl ;
    cout << "is_backend " << is_backend << endl ;


    if(is_frontend)
    {
        G4DAESocket<G4DAEArray> sock(frontend) ;
        G4DAEPhotonList* gpl = G4DAEPhotonList::Load("gdct001");
        gpl->Print();

        cout << "frontend " << frontend << " sending " << endl ; 
   
        //sock.SendObject((G4DAEArray*)gpl);
        sock.SendObject(gpl);

        G4DAEPhotonList* obj = (G4DAEPhotonList*)sock.ReceiveObject();
        obj->Print();

        //const char* msg = "hello";
        //sock.SendString((char*)msg);
        //const char* rep = sock.ReceiveString();

    }
    else if(is_backend)
    {
        char responder = 'P' ;
        G4DAESocket<G4DAEArray> sock(backend,responder) ;
        cout << "backend " << backend << " waiting " << endl ; 
        //sock.MirrorString();
        sock.MirrorObject();
    } 

    return 0 ;
}



int gpl_buffer()
{

    G4DAEPhotonList* gpl = G4DAEPhotonList::Load("gdct001");
    if(!gpl) return 1 ; 

    gpl->Print();
    gpl->Details(0);
    gpl->DumpBuffer();

    gpl->SaveToBuffer();



    return 0 ;
}




int main(int argc, char** argv){
    return gpl_buffer();
}




