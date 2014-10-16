#include <stdio.h>  
#include <stdlib.h>    

#include "G4GDMLParser.hh"
#include "G4DAEChroma/G4DAEChroma.hh"
#include "TFile.h"

//#include "Chroma/ChromaPhotonList.hh"
#include "ChromaPhotonList.hh"


int main(int argc, char** argv)
{
   const char* geokey = "DAE_NAME_DYB_GDML" ;
   const char* geopath = getenv(geokey);
   if(geopath == NULL ){
      printf("geokey %s : missing : use \"export-;export-export\" to define  \n", geokey );
      return 1;
   }   
   printf("geokey %s geopath %s \n", geokey, geopath ); 


   ChromaPhotonList* cpl = ChromaPhotonList::Load("1");
   if(cpl){
      cpl->Print();
   }


  /*
   G4GDMLParser fParser ; 
   fParser.Read(geopath,false);
   G4VPhysicalVolume* wpv = fParser.GetWorldVolume();       
   G4DAEChroma::GetG4DAEChroma()->CreateTransformCache(wpv); 
   G4DAEChroma::GetG4DAEChroma()->DumpTransformCache(); 
  */
   



   return 0 ; 
}

