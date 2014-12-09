#include "G4DAEChroma/G4DAEPhotons.hh"
#include "G4ThreeVector.hh"
#include <cstdlib>
#include <string.h>

#include <string>

#ifdef G4DAECHROMA_WITH_CPL
#include "G4DAEChroma/G4DAEChromaPhotonList.hh"
#else
class G4DAEChromaPhotonList ;
#endif

#include "G4DAEChroma/G4DAEPhotonList.hh"


const char* G4DAEPhotons::TMPL = "DAE_PATH_TEMPLATE" ; 
const char* G4DAEPhotons::KEY  = "NPL" ; 

// static utility funcs





void G4DAEPhotons::Transfer( G4DAEPhotons* dest , G4DAEPhotons* src, size_t a, size_t b )
{
   size_t nphoton = src->GetCount();

   G4ThreeVector pos ;
   G4ThreeVector mom ; 
   G4ThreeVector pol ; 
   float _t ; 
   float _wavelength ; 
   int _pmtid ;

   size_t skip=0, take=0 ;
   for(size_t index=0 ; index < nphoton ; ++index){
       if((b - a) > 0 && ( index < a || index >= b)){
             skip++ ;
             continue ;      // python style range 0:1 => [0]   0:2 => [0,1]
       } else {
             take++ ;
       }
       src->GetPhoton(index, pos, mom, pol, _t, _wavelength, _pmtid);
       dest->AddPhoton(pos, mom, pol, _t, _wavelength, _pmtid);
   }

   if((b - a) > 0)
   {
       printf("G4DAEPhotons::Transfer range selection a %zu b %zu selected %zu skipped %zu from source total %zu \n", a, b, take, skip, nphoton); 
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
#ifdef G4DAECHROMA_WITH_CPL
       photons = (G4DAEPhotons*)G4DAEChromaPhotonList::Load( name, key, tmpl );
#else
       printf("G4DAEPhotons::Load not compiled with support for deprecated ChromaPhotonList format \n");
       photons = NULL ;
#endif
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


G4DAEPhotons* G4DAEPhotons::LoadPath(const char* path, const char* key )
{   
   G4DAEPhotons* photons = NULL ; 

   if(HasExt(path, ".root"))
   {
#ifdef G4DAECHROMA_WITH_CPL
      photons = (G4DAEPhotons*)G4DAEChromaPhotonList::LoadPath( path, key );
#else
      printf("G4DAEPhotons::LoadPath not compiled with support for deprecated ChromaPhotonList format \n");
      photons = NULL ;
#endif
   } 
   else if(HasExt(path, ".npy"))
   {
      photons = (G4DAEPhotons*)G4DAEPhotonList::LoadPath( path, key );
   } 
   else 
   {
      printf("G4DAEPhotons::LoadPhotons unexpected file extension for path %s \n", path );
   }
   return photons ; 
}


#ifdef G4DAECHROMA_WITH_CPL
void G4DAEPhotons::SavePath(G4DAEChromaPhotonList* photons, const char* path, const char* key )
{
   if(!photons || !path) return ; 
   
   if(HasExt(path, ".root"))
   {
      printf("SavePath %s \n", path );
      photons->SavePath(path, key );
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
#endif

void G4DAEPhotons::SavePath(G4DAEPhotonList* photons, const char* path, const char* key )
{
   if(!photons || !path) return ; 
   
   if(HasExt(path, ".npy"))
   {
      printf("SavePath %s \n", path );
      photons->SavePath(path, key );
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
#ifdef G4DAECHROMA_WITH_CPL
   G4DAEChromaPhotonList* gcpl = dynamic_cast<G4DAEChromaPhotonList*>(photons); 
#else   
   G4DAEChromaPhotonList* gcpl = NULL ;
#endif

   if( gnpl )
   {
#ifdef VERBOSE
       printf("G4DAEPhotons::Save using G4DAEPhotonList \n");
#endif
       gnpl->Save( name, "NPL", tmpl);
   }
   else if( gcpl )
   {
#ifdef VERBOSE
       printf("G4DAEPhotons::Save using G4DAEChromaPhotonList \n");
#endif
#ifdef G4DAECHROMA_WITH_CPL
       gcpl->Save( name, "CPL", tmpl);
#else
       printf("G4DAEPhotons::Save not compiled to support deprecated G4DAEChromaPhotonList \n");
#endif
   } 
   else
   {
       printf("G4DAEPhotons::Save failed to dynamic_cast photons to allowed type \n");
   }
}








