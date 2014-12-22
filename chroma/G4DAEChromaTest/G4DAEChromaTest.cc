#include <stdio.h>  
#include <stdlib.h>    


#ifdef G4DAECHROMA_WITH_CPL
#include "G4DAEChroma/G4DAEChromaPhotonList.hh"
#endif

#include "G4DAEChroma/G4DAEFotonList.hh"
#include "G4DAEChroma/G4DAEScintillationStepList.hh"
#include "G4DAEChroma/G4DAECerenkovStepList.hh"

#include "G4DAEChroma/G4DAEPhotonList.hh"

#include "G4DAEChroma/G4DAESocketBase.hh"
#include "G4DAEChroma/G4DAECommon.hh"
#include "G4DAEChroma/G4DAEMetadata.hh"
#include "G4ThreeVector.hh"

#include <sstream>
#include <iostream>
#include <vector>

using namespace std ;

const char* frontend = "FRONTEND" ; 
const char* backend  = "BACKEND" ;



int gpl_string_network()
{

    if(getenv(frontend))
    {
        cout << __func__ << " " << frontend << " Send/Recv String" << endl ; 
        G4DAESocketBase sock(frontend) ;

        const char* request = __func__ ;

        sock.SendString((char*)request);
        const char* response = sock.ReceiveString();

        cout << __func__ << " " << "received " << response << endl ; 

    }
    else if(getenv(backend))
    {
        cout << __func__ << " " << backend << " MirrorString" << endl ; 
        G4DAESocketBase sock(backend,'P') ;
        sock.MirrorString();
    }
    return 0 ;
}



template<typename T>
int p_network()
{
    G4DAESocketBase* socket(NULL) ; 
    T* request(NULL) ;
    T* response(NULL) ;

    if(getenv(frontend))
    {
        socket = new G4DAESocketBase(frontend);

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



int metadata()
{
   G4DAEMetadata* meta = new G4DAEMetadata();

   delete meta ;
   return 0 ;
}


int test_array_growth(const char* evtkey)
{
   G4DAEPhotons* p = G4DAEPhotons::Load(evtkey);
   p->Print("test_array_growth");


   G4ThreeVector pos(3,3,3);
   G4ThreeVector dir(0,0,1);
   G4ThreeVector pol(0,0,1);

   float time = 1.; 
   float wavelength = 550.; 
   int pmtid = 0x1010101 ;

   size_t grow = 1000000 ; 
   for(size_t i=0 ; i < grow ; i++ ) p->AddPhoton( pos, dir, pol, time, wavelength, pmtid );
   p->Print("after growing");
   //p->Details(1);

   const char* xkey = "1x" ; 
   G4DAEPhotons::Save(p, xkey);

   G4DAEPhotons* x = G4DAEPhotons::Load(xkey);
   x->Print("after serialization/deserialization");

   return 0 ; 
}




int test_scintillationsteplist(const char* evtkey)
{
   G4DAEScintillationStepList* a = G4DAEScintillationStepList::Load(evtkey);
   a->Print("scintillationsteplist");
   return 0 ;
}

int test_cerenkovsteplist(const char* evtkey)
{
   G4DAECerenkovStepList* a = G4DAECerenkovStepList::Load(evtkey);
   a->Print("cerenkovsteplist");
   return 0 ;
}

int test_fotonlist(const char* evtkey)
{
   G4DAEFotonList* a = G4DAEFotonList::Load(evtkey);
   a->Print("fotonlist");
   return 0 ;
}



int main(int argc, char** argv)
{
    const char* evtkey = "1" ;

    test_fotonlist(evtkey);
    test_cerenkovsteplist(evtkey);
    test_scintillationsteplist(evtkey);


  /*
    test_array_growth(evtkey);



    p_copy<G4DAEChromaPhotonList,G4DAEPhotonList>(evtkey);

    p_copy<G4DAEChromaPhotonList,G4DAEChromaPhotonList>(evtkey);

    p_copy<G4DAEPhotonList,G4DAEChromaPhotonList>(evtkey);

    p_copy<G4DAEPhotonList,G4DAEPhotonList>(evtkey);


    p_save<G4DAEPhotonList>(evtkey);
    p_load<G4DAEPhotonList>(evtkey);

    p_save<G4DAEChromaPhotonList>(evtkey);
    p_load<G4DAEChromaPhotonList>(evtkey);

    p_buffer<G4DAEPhotonList>(evtkey);
    p_buffer<G4DAEChromaPhotonList>(evtkey);

  */

    // p_network<G4DAEPhotonList>();
    // p_network<G4DAEChromaPhotonList>();
    //p_string_network();

    //metadata();

    return 0 ;
}




