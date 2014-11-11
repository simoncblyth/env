#include "G4DAEChroma/G4DAEPhotons.hh"
#include "G4ThreeVector.hh"
#include <cstdlib>
#include <string.h>

#include <string>

#include "G4DAEChroma/G4DAEChromaPhotonList.hh"
#include "G4DAEChroma/G4DAEPhotonList.hh"

// static utility funcs

void G4DAEPhotons::Transfer( G4DAEPhotons* dest , G4DAEPhotons* src )
{
   size_t nphoton = src->GetPhotonCount();

   G4ThreeVector pos ;
   G4ThreeVector mom ; 
   G4ThreeVector pol ; 
   float _t ; 
   float _wavelength ; 
   int _pmtid ;

   for(size_t index=0 ; index < nphoton ; ++index){
       src->GetPhoton(index, pos, mom, pol, _t, _wavelength, _pmtid);
       dest->AddPhoton(pos, mom, pol, _t, _wavelength, _pmtid);
   }   
}

bool G4DAEPhotons::HasExt(const char* path, const char* ext)
{
   int plen = strlen(path);
   const char* pext = path + plen - strlen(ext)  ;
   return strcmp( pext, ext) == 0 ;
}

std::string G4DAEPhotons::SwapExt(const char* path, const char* aext, const char* bext)
{
   std::string ret ;
   if(!HasExt(path, aext)) return ret ;

   std::string swap(path, path+strlen(path)-strlen(aext));
   return swap + bext  ; 
}


G4DAEPhotons* G4DAEPhotons::Load(const char* name, const char* key, const char* tmpl)
{
   G4DAEPhotons* photons = NULL ; 
   if( strcmp( key, "CPL" ) == 0 )
   {
       photons = (G4DAEPhotons*)G4DAEChromaPhotonList::Load( name, key, tmpl );
   } 
   else if ( strcmp( key, "NPL" ) == 0 )
   {
       photons = (G4DAEPhotons*)G4DAEPhotonList::Load( name, key, tmpl );
   }
   else
   {
       printf("G4DAEPhotons::Load expects key to be either CPL or NPL not [%s] \n", key ); 
   }
   return photons ;
}


G4DAEPhotons* G4DAEPhotons::LoadPath(const char* path, const char* _key )
{   
   G4DAEPhotons* photons = NULL ; 

   if(HasExt(path, ".root"))
   {
      //printf("found ROOT %s \n", path );
      photons = (G4DAEPhotons*)G4DAEChromaPhotonList::LoadPath( path, _key );
   } 
   else if(HasExt(path, ".npy"))
   {
      //printf("found NPY  %s \n", path );
      photons = (G4DAEPhotons*)G4DAEPhotonList::LoadPath( path, _key );
   } 
   else 
   {
      printf("G4DAEPhotons::LoadPhotons unexpected file extension for path %s \n", path );
   }

   return photons ; 
}


void G4DAEPhotons::SavePath(G4DAEChromaPhotonList* photons, const char* path, const char* _key )
{
   if(!photons || !path) return ; 
   
   if(HasExt(path, ".root"))
   {
      printf("SavePath %s \n", path );
      photons->SavePath(path, _key );
   } 
   else if(HasExt(path, ".npy"))
   {
      printf(".npy is the wrong extension for persisting G4DAEChromaPhotonList  %s \n", path );
   } 
   else 
   {
      printf("unexpected file extension for path %s \n", path );
   }
}

void G4DAEPhotons::SavePath(G4DAEPhotonList* photons, const char* path, const char* _key )
{
   if(!photons || !path) return ; 
   
   if(HasExt(path, ".npy"))
   {
      printf("SavePath %s \n", path );
      photons->SavePath(path, _key );
   } 
   else if(HasExt(path, ".root"))
   {
      printf(".root is the wrong extension for persisting G4DAEPhotonList  %s \n", path );
   } 
   else 
   {
      printf("unexpected file extension for path %s \n", path );
   }
}


void G4DAEPhotons::Save(G4DAEPhotons* photons, const char* name, const char* /*key*/, const char* tmpl )
{
   if(!photons || !name || !tmpl) return ; 
   
    // distinguish the flavor of photons by dynamic casting 
   G4DAEPhotonList* gnpl = dynamic_cast<G4DAEPhotonList*>(photons);
   G4DAEChromaPhotonList* gcpl = dynamic_cast<G4DAEChromaPhotonList*>(photons); 

   if( gnpl )
   {
       printf("G4DAEPhotons::Save using G4DAEPhotonList \n");
       gnpl->Save( name, "NPL", tmpl);
   }
   else if( gcpl )
   {
       printf("G4DAEPhotons::Save using G4DAEChromaPhotonList \n");
       gcpl->Save( name, "CPL", tmpl);
   } 
   else
   {
       printf("failed to dynamic_cast photons to allowed type \n");
   }
}








