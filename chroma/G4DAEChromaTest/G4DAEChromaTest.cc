#include <stdio.h>  
#include <stdlib.h>    

//#include "Chroma/ChromaPhotonList.hh"
//#include "ChromaPhotonList.hh"
#include "G4DAEChroma/G4DAEGeometry.hh"
#include "G4DAEChroma/G4DAEChroma.hh"

int main(int argc, char** argv)
{
   const char* geokey = "DAE_NAME_DYB_GDML";

  /*
   const char* evtkey = "1" ;
   ChromaPhotonList* cpl = ChromaPhotonList::Load(evtkey);
   if(!cpl){
       printf("failed to load photons with evtkey %s \n", evtkey);
       return 1 ;
   }
   //cpl->Print();
   */


   G4DAEGeometry* geo = G4DAEGeometry::LoadFromGDML(geokey);
   //G4DAEGeometry* geo = NULL ;
   if(!geo){
       printf("failed to load geometry with geokey %s \n", geokey);
       return 1 ;
   }
   //geo->DumpTransformCache();


   G4DAEChroma* gdc = G4DAEChroma::GetG4DAEChroma();
   gdc->SetGeometry(geo);
   //gdc->ProcessHit( cpl, 0 );

   return 0 ; 
}

