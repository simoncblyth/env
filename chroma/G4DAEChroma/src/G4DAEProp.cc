#include "G4DAEChroma/G4DAEProp.hh"
#include "G4DAEChroma/G4DAEArrayHolder.hh" 

#include "G4PhysicsOrderedFreeVector.hh" 

const char* G4DAEProp::TMPL = "DAE_PROP_PATH_TEMPLATE" ;
const char* G4DAEProp::SHAPE = "2" ;
const char* G4DAEProp::KEY   = "PRP" ;

G4DAEArrayHolder* G4DAEProp::Copy(G4PhysicsOrderedFreeVector* pofv)
{
    size_t size = pofv->GetVectorLength() ;
    G4DAEArrayHolder* holder = new G4DAEArrayHolder( size, NULL, "2" );
    for(int b=0 ; b < size ; ++b)
    {
       float* prop = holder->GetNextPointer(); 
       prop[_binEdge]  = pofv->GetLowEdgeEnergy(b);
       prop[_binValue] = (*pofv)[b] ; 
    }
    return holder ; 
}





