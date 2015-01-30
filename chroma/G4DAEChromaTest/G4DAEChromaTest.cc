#include <stdio.h>  
#include <stdlib.h>    


#ifdef G4DAECHROMA_WITH_CPL
#include "G4DAEChroma/G4DAEChromaPhotonList.hh"
#endif

#include "G4DAEChroma/G4DAEChroma.hh"
#include "G4DAEChroma/G4DAEPhotonList.hh"
#include "G4DAEChroma/G4DAECerenkovPhotonList.hh"
#include "G4DAEChroma/G4DAEScintillationPhotonList.hh"
#include "G4DAEChroma/G4DAEScintillationStepList.hh"
#include "G4DAEChroma/G4DAECerenkovStepList.hh"


#include "G4DAEChroma/G4DAESocketBase.hh"
#include "G4DAEChroma/G4DAECommon.hh"
#include "G4DAEChroma/G4DAEMetadata.hh"
#include "G4DAEChroma/G4DAEPropList.hh"



#include "Randomize.hh"

#include "G4ThreeVector.hh"
#include "G4PhysicsOrderedFreeVector.hh"


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
   G4DAEPhotonList* p = G4DAEPhotonList::Load(evtkey);
   p->Print("test_array_growth");


   G4ThreeVector pos(3,3,3);
   G4ThreeVector dir(0,0,1);
   G4ThreeVector pol(0,0,1);

   float time = 1.; 
   float wavelength = 550.; 
   int pmtid = 0x1010101 ;

   size_t grow = 1000000 ; 
   //for(size_t i=0 ; i < grow ; i++ ) p->AddPhoton( pos, dir, pol, time, wavelength, pmtid );
   p->Print("after growing");
   //p->Details(1);

   const char* xkey = "1x" ; 
   p->Save(xkey);

   G4DAEPhotonList* x = G4DAEPhotonList::Load(xkey);
   x->Print("after serialization/deserialization");

   return 0 ; 
}




int test_scintillationstep_list(const char* evtkey)
{
   G4DAEScintillationStepList* a = G4DAEScintillationStepList::Load(evtkey);
   a->Print("scintillationsteplist");
   return 0 ;
}

int test_cerenkovstep_list(const char* evtkey)
{
   G4DAECerenkovStepList* a = G4DAECerenkovStepList::Load(evtkey);
   a->Print("cerenkovsteplist");
   return 0 ;
}

int test_foton_list(const char* evtkey)
{
   G4DAEScintillationPhotonList* a = G4DAEScintillationPhotonList::Load(evtkey);
   a->Print("fotonlist");
   return 0 ;
}

int test_photon_list(const char* evtkey)
{
   G4DAEPhotonList* a = G4DAEPhotonList::Load(evtkey);
   a->Print("photonlist");
   return 0 ;
}

int test_generate(const char* evtkey )
{
   G4DAECerenkovStepList* a = G4DAECerenkovStepList::Load(evtkey);
   a->Print("cerenkovsteplist");

   G4DAESocketBase* socket = new G4DAESocketBase(frontend);

   G4DAEArrayHolder* response = socket->SendReceive(a);
   if(!response) return 1 ; 

   printf("response %p \n", (void*)response);
   response->DumpBuffer();
   response->Print("response");


   // NB the below do not have their own storage, they 
   // are just different "view"s of the response 

   G4DAEScintillationPhotonList* fl = new G4DAEScintillationPhotonList(response) ; 
   fl->Print("fotonlist");
   fl->Save("test"); 

   G4DAECerenkovPhotonList* xl = new G4DAECerenkovPhotonList(response) ; 
   xl->Print("xotonlist");
   xl->Save("test"); 

   G4DAEPhotonList* pl = new G4DAEPhotonList(response) ; 
   pl->Print("photonlist");
   pl->Save("test"); 

   G4DAECerenkovStepList* cl = new G4DAECerenkovStepList(response) ; 
   cl->Print("cerenkovsteplist");
   cl->Save("test"); 

   G4DAEScintillationStepList* sl = new G4DAEScintillationStepList(response) ; 
   sl->Print("scintillationsteplist");
   sl->Save("test"); 


   return 0 ;
}



void dump_POFV(G4PhysicsOrderedFreeVector* pofv)
{
    cout << "MaxValue         " << pofv->GetMaxValue() << endl ;
    cout << "MinValue         " << pofv->GetMinValue() << endl ;
    cout << "MaxLowEdgeEnergy " << pofv->GetMaxLowEdgeEnergy() << endl ;
    cout << "MaxMinEdgeEnergy " << pofv->GetMinLowEdgeEnergy() << endl ;
    cout << "VectorLength     " << pofv->GetVectorLength() << endl ;
}


int test_G4DAEPropList()
{

    G4PhysicsOrderedFreeVector* pofv = new G4PhysicsOrderedFreeVector();
    float nm = 80. ;
    while( nm < 800. )
    { 
        pofv->InsertValues( nm, nm*10. );
        nm += 20.5 ;
    } 

    pofv->DumpValues();
    dump_POFV(pofv);

    G4DAEPropList a(G4DAEProp::Copy(pofv));
    a.Save("check");

    return 0 ;
} 
 

int test_G4DAEPropList_read()
{
    G4DAEPropList* pl = G4DAEPropList::Load("gdls_fast");
    pl->Print();
   
    G4PhysicsOrderedFreeVector* pofv = G4DAEProp::CreatePOFV(pl);

    pofv->DumpValues();
    dump_POFV(pofv);
    
    return 0 ;
}


int test_ScintillationIntegral()
{
    G4DAEPropList* cdf = G4DAEPropList::Load("gdls_fast");
    cdf->Print();
    G4PhysicsOrderedFreeVector* ScintillationIntegral = G4DAEProp::CreatePOFV(cdf);
    G4double MaxValue = ScintillationIntegral->GetMaxValue() ;

    //size_t size = 1e6 ; 
    size_t size = 2817543 ;  // match the count to current evt "1"

    G4DAEArrayHolder* holder = new G4DAEArrayHolder( size, NULL, "2" );
    for(size_t n=0 ; n<size ; n++ )
    {
        G4double CIIvalue = G4UniformRand()*MaxValue;
        G4double sampledEnergy = ScintillationIntegral->GetEnergy(CIIvalue);

        float* prop = holder->GetNextPointer();
        prop[G4DAEProp::_binEdge]  = float(CIIvalue) ;
        prop[G4DAEProp::_binValue] = float(sampledEnergy) ;
    }

    G4DAEPropList dist(holder); 
    dist.Save("1");  // sampledEnergy

    //
    //  cf('wavelength', typs="gopscintillation opscintillation prop",tag=1,  log=True, range=(100,900) )
    //   succeeds to match G4 Scintillation photon wavelength distrib 
    //
    return 0 ; 
}


 

int test_G4DAEChroma_flags()
{

    G4DAEChroma* chroma = G4DAEChroma::GetG4DAEChroma(); 

    //string flags = "FLAG_G4SCINTILLATION_COLLECT_STEP,FLAG_G4CERENKOV_COLLECT_STEP," ;
    string flags = "FLAG_G4SCINTILLATION_KILL_SECONDARY,FLAG_G4CERENKOV_KILL_SECONDARY," ;

    chroma->SetFlags(flags);

    cout << chroma->Flags() << endl ;    

    return 0 ; 
}


int test_G4DAEMetadata()
{
    const char* jspath = getenv("G4DAECHROMA_CONFIG_PATH");
    if(!jspath) return 1 ; 

    G4DAEMetadata* meta = G4DAEMetadata::CreateFromFile(jspath);
    meta->Print();
    Map_t map = meta->GetRawMap("/FLAGS");
    for(Map_t::iterator it=map.begin() ; it != map.end() ; it++ )
    {   
        string key = it->first ;
        string val = it->second ;
        int ival = atoi(val.c_str());
        printf(" %20s : %s : %d \n", key.c_str(), val.c_str(), ival );
    }   

    return 0 ; 
}


int test_G4DAEManager()
{
    G4DAEChroma* mgr = G4DAEChroma::GetG4DAEChroma();
    return 0 ; 
}


void test_G4DAECommon_removeField()
{
     const char* token = "token_to_be_removed" ;
     const char* line = "/path/to/geometry.dae.noextra.token_to_be_removed.dae" ;

     std::string path = removeField(line, '.', -2 );
     printf("line %s \n", line);
     printf("path %s \n", path.c_str());

     std::string chk = insertField( path.c_str(), '.', -1, token );
     printf("chk  %s \n", chk.c_str());

     assert(strcmp(line, chk.c_str()) == 0);

}

int main(int argc, char** argv)
{

    //test_G4DAEManager();

    //test_G4DAEMetadata();
    //test_G4DAEChroma_flags();

    test_G4DAECommon_removeField();
   

    //test_ScintillationIntegral();
    //test_G4DAEPropList_read();
    //test_G4DAEPropList();

    //const char* evtkey = "1" ;
    //test_generate(evtkey);


  /*
    test_foton_list(evtkey);
    test_cerenkovstep_list(evtkey);
    test_scintillationstep_list(evtkey);
    test_photon_list(evtkey);
  */

  /*
    test_array_growth(evtkey);
  */


  /*

    p_copy<G4DAEChromaPhotonList,G4DAEPhotonList>(evtkey);

    p_copy<G4DAEChromaPhotonList,G4DAEChromaPhotonList>(evtkey);

    p_copy<G4DAEPhotonList,G4DAEChromaPhotonList>(evtkey);

    p_copy<G4DAEPhotonList,G4DAEPhotonList>(evtkey);
 */ 


  /*
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




