#include <stdio.h>  
#include <stdlib.h>    

//#include "Chroma/ChromaPhotonList.hh"
#include "G4DAEChroma/G4DAEChromaPhotonList.hh"
#include "G4DAEChroma/G4DAEPhotonList.hh"

//#include "G4DAEChroma/G4DAESocket.hh"
#include "G4DAEChroma/G4DAESocketBase.hh"
#include "G4DAEChroma/G4DAECommon.hh"
#include "G4ThreeVector.hh"

#include <sstream>
#include <iostream>
#include <vector>

using namespace std ;

const char* frontend = "FRONTEND" ; 
const char* backend  = "BACKEND" ;


//typedef G4DAESocket<G4DAEArray> G4DAESocket_t ;
//typedef G4DAESocket<G4DAEPhotonList> G4DAESocket_t ;

// non-template variant dealing in G4DAESerializable types
typedef G4DAESocketBase G4DAESocket_t ;   


// **SUCH TYPEDEF OK IN TEST CODE : TOO CONFUSING IN MORE SOLID CODE**
typedef G4DAEChromaPhotonList G4DAEPhotons_t ;  // .root TObject serialization 
//typedef G4DAEPhotonList       G4DAEPhotons_t ;    // .npy NPY serialization 

/*
   Running into bad access trouble with ROOT serialization , 
   need to make dict at higher level G4DAEChromaPhotonList rather than 
   ChromaPhotonList to get it properly persisted
*/



int gpl_string_network()
{

    if(getenv(frontend))
    {
        cout << __func__ << " " << frontend << " Send/Recv String" << endl ; 
        G4DAESocket_t sock(frontend) ;

        const char* request = __func__ ;

        sock.SendString((char*)request);
        const char* response = sock.ReceiveString();

        cout << __func__ << " " << "received " << response << endl ; 

    }
    else if(getenv(backend))
    {
        cout << __func__ << " " << backend << " MirrorString" << endl ; 
        G4DAESocket_t sock(backend,'P') ;
        sock.MirrorString();
    }
    return 0 ;
}



template<typename T>
int p_network()
{
    G4DAESocket_t* socket(NULL) ; 
    T* request(NULL) ;
    T* response(NULL) ;

    if(getenv(frontend))
    {
        socket = new G4DAESocket_t(frontend);

        request = T::Load("1");

        response = reinterpret_cast<T*>(socket->SendReceiveObject(request));


        if( request->GetPhotonDigest() == response->GetPhotonDigest() ){
            cout << "request and response digests match " << endl ; 
        } else {
            cout << "request and response digests differ " << endl ; 
            request->Print();
            response->Print();
            request->Details(0);
            response->Details(0);
        }

    }
    else if(getenv(backend))
    {
        cout << __func__ << " " << backend << " MirrorObject" << endl ; 
        socket = new G4DAESocketBase(backend, 'P');
        socket->MirrorObject();
    } 
    else
    {
        cout << "need to define FRONTEND or BACKEND envvars " << endl ;
    }

    delete socket ; 
    delete request ; 
    delete response  ; 

    return 0 ;
}



template<typename T>
int p_buffer(const char* evtkey)
{

    T* p = T::Load(evtkey);
    if(!p) return 1 ; 

    p->Print();

    printf("p_buffer SaveToBuffer\n");
    p->SaveToBuffer();

    printf("p_buffer DumpBuffer\n");
    p->DumpBuffer();

    printf("p_buffer Details\n");
    p->Details(0);


    return 0 ;
}

template<typename T>
int p_save(const char* evtkey)
{
    T* p = new T(5);

    G4ThreeVector pos(3,3,3);
    G4ThreeVector dir(0,0,1);
    G4ThreeVector pol(0,0,1);

    float time = 1.; 
    float wavelength = 550.; 
    int pmtid = 0x1010101 ;

    p->AddPhoton( pos, dir, pol, time, wavelength, pmtid );
    p->AddPhoton( pos, dir, pol, time, wavelength, pmtid );
    p->AddPhoton( pos, dir, pol, time, wavelength, pmtid );

    p->Save(evtkey);

    delete p ;  
    return 0;
}

template<typename T>
int p_load(const char* evtkey)
{
   // NB due to the default templates having file exts 
   // .root and .npy the 2 different photon serializations 
   // load/save the corresponding files
   // 
   T* p = T::Load(evtkey);

   if(!p){
       printf("failed to load photons with evtkey %s \n", evtkey);
   }
   p->Print();
   p->Details(0);
   return 0 ; 
}


template<typename A, typename B>
int p_copy(const char* evtkey)
{
   A* a = A::Load(evtkey);
   a->Print();
   a->Details(0);

   B* b = new B(a);
   b->Print();
   b->Details(0);

   return 0 ;
}




int main(int argc, char** argv)
{
    const char* evtkey = "3" ;
    p_copy<G4DAEChromaPhotonList,G4DAEPhotonList>(evtkey);
    p_copy<G4DAEChromaPhotonList,G4DAEChromaPhotonList>(evtkey);
    p_copy<G4DAEPhotonList,G4DAEChromaPhotonList>(evtkey);
    p_copy<G4DAEPhotonList,G4DAEPhotonList>(evtkey);

    //p_network<G4DAEPhotonList>();
    //p_network<G4DAEChromaPhotonList>();
    //p_string_network();

    /*

    p_save<G4DAEPhotonList>(evtkey);
    p_load<G4DAEPhotonList>(evtkey);

    p_save<G4DAEChromaPhotonList>(evtkey);
    p_load<G4DAEChromaPhotonList>(evtkey);

    p_buffer<G4DAEPhotonList>(evtkey);
    p_buffer<G4DAEChromaPhotonList>(evtkey);


    */

    return 0 ;
}




