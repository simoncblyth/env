#include "G4DAEChroma/G4DAEProp.hh"
#include "G4DAEChroma/G4DAEArrayHolder.hh" 

#include "G4PhysicsOrderedFreeVector.hh" 

const char* G4DAEProp::TMPL = "DAE_PROP_PATH_TEMPLATE" ;
const char* G4DAEProp::SHAPE = "2" ;
const char* G4DAEProp::KEY   = "PRP" ;

#include <iostream>
using namespace std ; 

G4DAEArrayHolder* G4DAEProp::Copy(G4PhysicsOrderedFreeVector* pofv, double xscale, double yscale )
{
    size_t size = pofv->GetVectorLength() ;
    printf("G4DAEProp::Copy size %zu \n", size ); 
    G4DAEArrayHolder* holder = new G4DAEArrayHolder( size, NULL, "2" );
    for(size_t b=0 ; b < size ; ++b)
    {
       float* prop = holder->GetNextPointer();

       double d_edge = pofv->GetLowEdgeEnergy(b)*xscale ;
       double d_value  = (*pofv)[b]*yscale ; 

       prop[_binEdge]  = float(d_edge) ;
       prop[_binValue] = float(d_value) ; 
    }

    
    return holder ; 
}


G4PhysicsOrderedFreeVector* G4DAEProp::CreatePOFV(G4DAEArrayHolder* holder, double xscale, double yscale)
{
    G4PhysicsOrderedFreeVector* pofv = new G4PhysicsOrderedFreeVector();
    size_t size = holder->GetCount();
    printf("G4DAEProp::Create size %zu \n", size ); 
    for(size_t index=0 ; index < size ; index++ )
    {   
        float* prop = holder->GetItemPointer( index );
        float edge = prop[_binEdge] ;
        float value = prop[_binValue] ;

        double d_edge  = edge*xscale ; 
        double d_value = value*yscale ; 

        pofv->InsertValues(d_edge, d_value);
    }   
    return pofv ; 
}






