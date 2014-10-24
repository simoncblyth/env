#include <stdio.h>  
#include <stdlib.h>    

#include "Chroma/ChromaPhotonList.hh"
#include "G4DAEChroma/G4DAEGeometry.hh"
#include "G4DAEChroma/G4DAEChroma.hh"
#include "G4DAEChroma/G4DAESensDet.hh"
#include "G4DAEChroma/DemoG4DAECollector.hh"

#include "G4SDManager.hh"

#include <sstream>
#include <iostream>
#include <vector>

using namespace std ;

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

void configure()
{
   G4DAEChroma* chroma = G4DAEChroma::GetG4DAEChroma();

   const char* transport = "" ;
   const char* sensdet = "DsPmtSensDet" ;
   const char* geometry = "DAE_NAME_DYB_GDML" ;

   chroma->Configure( transport, sensdet, geometry );
   //chroma->ProcessHit( cpl, 0 );

}



int main(int argc, char** argv)
{

    G4SDManager* SDMan = G4SDManager::GetSDMpointer();
    SDMan->SetVerboseLevel( 10 );

    G4DAESensDet* sd = new G4DAESensDet("DsPmtSensDet");
    sd->SetCollector(new DemoG4DAECollector );  
    sd->initialize();
    SDMan->AddNewDetector( sd );

    const char* geometry = "DAE_NAME_DYB_GDML" ;
    G4DAEGeometry* geo = G4DAEGeometry::LoadFromGDML(geometry, sd);


    return 0 ; 
}

