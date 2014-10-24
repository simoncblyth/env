#include <stdio.h>  
#include <stdlib.h>    

#include "Chroma/ChromaPhotonList.hh"
#include "G4DAEChroma/G4DAEGeometry.hh"
#include "G4DAEChroma/G4DAEChroma.hh"


void loadgeom(const char* geometry)
{
   G4DAEGeometry* geo = G4DAEGeometry::LoadFromGDML(geometry);
   if(!geo){
       printf("failed to load geometry with geokey %s \n", geometry);
   }
   geo->DumpTransformCache();
}


void loadphotons(const char* evtkey)
{
   ChromaPhotonList* cpl = ChromaPhotonList::Load(evtkey);
   if(!cpl){
       printf("failed to load photons with evtkey %s \n", evtkey);
   }
   cpl->Print();
}

/*

   G4DAEChroma testing in bare environment


   no-network testing ? 


   * without 


*/



int main(int argc, char** argv)
{
   G4DAEChroma* chroma = G4DAEChroma::GetG4DAEChroma();

   const char* transport = "" ;
   const char* sensdet = "DsPmtSensDet" ;
   const char* geometry = "DAE_NAME_DYB_GDML" ;

   chroma->Configure( transport, sensdet, geometry );
   //chroma->ProcessHit( cpl, 0 );

   return 0 ; 
}

