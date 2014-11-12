#include "G4DAEChroma/G4DAEHitList.hh"
#include "G4DAEChroma/G4DAECommon.hh"
#include "G4DAEChroma/G4DAEArray.hh"

#include "numpy.hpp"
#include <cassert>
#include <sys/stat.h> 
#include <iostream>

using namespace std ; 


G4DAEHitList::G4DAEHitList( G4DAEArray* arr ) : m_array(arr) 
{
}

G4DAEHitList::G4DAEHitList( std::size_t itemcapacity, float* data) : m_array(NULL) 
{
    m_array = new G4DAEArray( itemcapacity, "8,3", data );
}


G4DAEHitList* G4DAEHitList::Load(const char* evt, const char* key, const char* tmpl )
{
    G4DAEArray* array = G4DAEArray::Load(evt, key, tmpl);
    return new G4DAEHitList(array) ;  
}

G4DAEHitList* G4DAEHitList::LoadPath(const char* path, const char* key )
{
    G4DAEArray* array = G4DAEArray::LoadPath(path, key );
    return new G4DAEHitList(array) ;  
}

void G4DAEHitList::Save(const char* evt, const char* key, const char* tmpl )
{
    if(m_array) m_array->Save(evt, key, tmpl);
}
void G4DAEHitList::SavePath(const char* path, const char* key)
{
    if(m_array) m_array->SavePath(path, key );
}

string G4DAEHitList::GetPath(const char* evt, const char* tmpl )
{
    return G4DAEArray::GetPath(evt, tmpl);
}

G4DAEHitList::~G4DAEHitList()
{
   delete m_array ; 
}


enum {
   gpos_x,
   gpos_y,
   gpos_z,

   gdir_x,
   gdir_y,
   gdir_z,

   gpol_x,
   gpol_y,
   gpol_z,

   lpos_x,
   lpos_y,
   lpos_z,

   ldir_x,
   ldir_y,
   ldir_z,

   lpol_x,
   lpol_y,
   lpol_z,

   t, 
   wavelength,
   weight,

   pmtid,
   trackid,
   spare

};




void G4DAEHitList::AddHit( G4DAEHit& hit )
{
    float* data = m_array->GetNextPointer();

    data[gpos_x] =  hit.gpos.x() ;
    data[gpos_y] =  hit.gpos.y() ;
    data[gpos_z] =  hit.gpos.z() ;

    data[gdir_x] =  hit.gdir.x() ;
    data[gdir_y] =  hit.gdir.y() ;
    data[gdir_z] =  hit.gdir.z() ;

    data[gpol_x] =  hit.gpol.x() ;
    data[gpol_y] =  hit.gpol.y() ;
    data[gpol_z] =  hit.gpol.z() ;

    data[lpos_x] =  hit.lpos.x() ;
    data[lpos_y] =  hit.lpos.y() ;
    data[lpos_z] =  hit.lpos.z() ;

    data[ldir_x] =  hit.ldir.x() ;
    data[ldir_y] =  hit.ldir.y() ;
    data[ldir_z] =  hit.ldir.z() ;

    data[lpol_x] =  hit.lpol.x() ;
    data[lpol_y] =  hit.lpol.y() ;
    data[lpol_z] =  hit.lpol.z() ;

    data[t]          =  hit.t ;
    data[wavelength] =  hit.wavelength ;
    data[weight]     =  hit.weight ;


    uif_t uifd[3] ; 

    uifd[0].i = hit.pmtid ;
    uifd[1].i = hit.trackid ;
    uifd[2].i = 0     ;

    data[pmtid]   =  uifd[0].f ;
    data[trackid] =  uifd[1].f ;
    data[spare]   =  uifd[2].f ;
    

}

