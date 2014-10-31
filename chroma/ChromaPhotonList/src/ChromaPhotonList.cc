#include "Chroma/ChromaPhotonList.hh"
#include "assert.h"
#include <iostream>

#include "TMD5.h"
#include "TFile.h"

using namespace std ; 


/*
ChromaPhotonList::Load
========================

Loads ChromaPhotonList objects from root files

Usage::

   ChromaPhotonList* cpl = ChromaPhotonList::Load();
   ChromaPhotonList* cpl = ChromaPhotonList::Load("1","CPL","DAE_PATH_TEMPLATE");
   ChromaPhotonList* cpl = ChromaPhotonList::Load("20140514-174932");

The tmpl envvar is expected to have a single %s field which 
is filled by the first *evt* argument eg  

    DAE_PATH_TEMPLATE=/usr/local/env/tmp/%s.root

The *key* argument is used for TFile loading 

*/

string ChromaPhotonList::GetPath( const char* evt , const char* tmpl )
{
   string empty ;
   const char* evtfmt  = getenv(tmpl);
   if(evtfmt == NULL ){
      printf("tmpl %s : missing : use \"export-;export-export\" to define  \n", tmpl );
      return empty; 
   }   
   char evtpath[256];
   if (sprintf(evtpath, evtfmt, evt ) < 0) return empty;
   return string( evtpath );
}


ChromaPhotonList* ChromaPhotonList::Load(const char* evt, const char* evtkey, const char* tmpl)
{
   string evtpath = GetPath(evt, tmpl);
   if( evtpath.empty() )
   {
      printf("ChromaPhotonList::Load : failed to format evtpath from tmpl  %s and evt %s \n", tmpl, evt );  
      return NULL ; 
   } 

   TFile fevt( evtpath.c_str(), "READ" );
   if( fevt.IsZombie() ){
       cout << "ChromaPhotonList::Load : failed to open evtpath [" << evtpath << "]" << endl ;
       return NULL ; 
   }   

   TObject* obj = fevt.Get(evtkey);
   ChromaPhotonList* cpl = (ChromaPhotonList*)obj ;
   return cpl ; 
}

void ChromaPhotonList::Save(const char* evt, const char* evtkey, const char* tmpl)
{
   string evtpath = GetPath(evt, tmpl);
   if( evtpath.empty() )
   {
      printf("ChromaPhotonList::Save : failed to format evtpath from tmpl  %s and evt %s \n", tmpl, evt );  
      return ; 
   } 

   TFile fevt( evtpath.c_str(), "RECREATE", evtkey );
   if( fevt.IsZombie() ){
       cout << "ChromaPhotonList::Save : failed to open evtpath for writing [" << evtpath << "]" << endl ;
   }   
   this->Write(evtkey);
   fevt.Close();

   cout << "ChromaPhotonList::Save : saved with key " << evtkey << " to path " << evtpath << endl ; 
}


ChromaPhotonList::ChromaPhotonList() : TObject() {
}

ChromaPhotonList::~ChromaPhotonList() {

}

void ChromaPhotonList::Print(Option_t* option) const 
{
    //  http://root.cern.ch/phpBB3/viewtopic.php?t=9837
    //  Rene: If you do not use TRef, TRefArray, you can freely use the TObject UniqueID.  
    //
    std::cout <<  "ChromaPhotonList::Print UID [" << GetUniqueID() << "]" << option << " [" << GetSize() << "]" << std::endl ;    
} 

std::size_t ChromaPhotonList::GetSize() const
{
   return x.size() ;  
}


std::string ChromaPhotonList::GetDigest() const 
{
  //const float* bufx = x.data(); 
  //for(size_t i=0 ; i < x.size() ; i++ ) if ( i < 10 || i > x.size() - 10 ) std::cout << i << " " << bufx[i] << std::endl ;

  // digest of all photon data by casting underlying contiguous arrays as bytes 
  TMD5 md5 ;
  md5.Update( (UChar_t*)x.data(), sizeof(float)*x.size() ); 
  md5.Update( (UChar_t*)y.data(), sizeof(float)*y.size() ); 
  md5.Update( (UChar_t*)z.data(), sizeof(float)*z.size() ); 
  md5.Update( (UChar_t*)px.data(), sizeof(float)*px.size() ); 
  md5.Update( (UChar_t*)py.data(), sizeof(float)*py.size() ); 
  md5.Update( (UChar_t*)pz.data(), sizeof(float)*pz.size() ); 
  md5.Update( (UChar_t*)polx.data(), sizeof(float)*polx.size() ); 
  md5.Update( (UChar_t*)poly.data(), sizeof(float)*poly.size() ); 
  md5.Update( (UChar_t*)polz.data(), sizeof(float)*polz.size() ); 
  md5.Update( (UChar_t*)t.data(), sizeof(float)*t.size() ); 
  md5.Update( (UChar_t*)wavelength.data(), sizeof(float)*wavelength.size() ); 

  md5.Update( (UChar_t*)pmtid.data(), sizeof(int)*pmtid.size() ); 
  md5.Final();
  return md5.AsString() ;
}



#ifdef CPL_WITH_GEANT4
void ChromaPhotonList::AddPhoton(G4ThreeVector pos, G4ThreeVector mom, G4ThreeVector pol, float _t, float _wavelength, int _pmtid) 
{
    x.push_back(pos.x());
    y.push_back(pos.y());
    z.push_back(pos.z());
    px.push_back(mom.x());
    py.push_back(mom.y());
    pz.push_back(mom.z());
    polx.push_back(pol.x());
    poly.push_back(pol.y());
    polz.push_back(pol.z());
    t.push_back(_t);
    wavelength.push_back(_wavelength);
    pmtid.push_back(_pmtid);
}
#endif


void ChromaPhotonList::AddPhoton(float _x, float _y, float _z,  
                                 float _px, float _py, float _pz, 
                                 float _polx, float _poly, float _polz, 
                                 float _t, float _wavelength, int _pmtid) 
{
    x.push_back(_x);
    y.push_back(_y);
    z.push_back(_z);
    px.push_back(_px);
    py.push_back(_py);
    pz.push_back(_pz);
    polx.push_back(_polx);
    poly.push_back(_poly);
    polz.push_back(_polz);
    t.push_back(_t);
    wavelength.push_back(_wavelength);
    pmtid.push_back(_pmtid);
}


void ChromaPhotonList::ClearAll() 
{
    x.clear();
    y.clear();
    z.clear();
    px.clear();
    py.clear();
    pz.clear();
    polx.clear();
    poly.clear();
    polz.clear();
    t.clear();
    wavelength.clear();
    pmtid.clear();
}


 
// Build a ChromaPhotonList object from C arrays
void ChromaPhotonList::FromArrays(float* __x,    float* __y,    float* __z,
                  float* __px,   float* __py,   float* __pz,
                  float* __polx, float* __poly, float* __polz,
                  float* __t, float* __wavelength, int* __pmtid, int nphotons) 
{
    for (int i=0; i<nphotons; i++) { 
      x.push_back(__x[i]);
      y.push_back(__y[i]);
      z.push_back(__z[i]);
      px.push_back(__px[i]);
      py.push_back(__py[i]);
      pz.push_back(__pz[i]);
      polx.push_back(__polx[i]);
      poly.push_back(__poly[i]);
      polz.push_back(__polz[i]);
      t.push_back(__t[i]);
      wavelength.push_back(__wavelength[i]);
      pmtid.push_back(__pmtid[i]);
    }
}



void ChromaPhotonList::Details(bool hit) const 
{
    cout <<  "ChromaPhotonList::Details [" << x.size() << "]" << endl ;

    float _t ;
    float _wavelength ;
    int _pmtid ;
    size_t index ; 

#ifdef CPL_WITH_GEANT4
    G4ThreeVector pos ;
    G4ThreeVector mom ;
    G4ThreeVector pol ;
    for( index = 0 ; index < x.size() ; index++ )
    {
        GetPhoton( index , pos, mom, pol, _t, _wavelength, _pmtid );    
        cout << " index " << index 
             << " pos " << pos 
             << " mom " << mom 
             << " pol " << pol 
             << " _t " << _t 
             << " _wavelength " << _wavelength 
             << " _pmtid " << (void*)_pmtid 
             << endl ; 
    }
#else
    float _x, _y, _z;
    float _px, _py, _pz;
    float _polx, _poly, _polz;
    for( index = 0 ; index < x.size() ; index++ )
    {
        GetPhoton( index, _x,_y,_z, _px,_py,_pz, _polx,_poly,_polz, _t, _wavelength, _pmtid );    

        if( ( hit && _pmtid > -1 ) || not hit ) cout << " index " << index 
                 << " pos " << _x << " " << _y << " " << _z  
                 << " mom " << _px << " " << _py << " " << _pz  
                 << " pol " << _polx << " " << _poly << " " << _polz 
                 << " _t " << _t 
                 << " _wavelength " << _wavelength 
                 << " _pmtid " << (void*)_pmtid 
                 << endl ; 
    }

#endif


} 


#ifdef CPL_WITH_GEANT4
void ChromaPhotonList::GetPhoton(size_t index, G4ThreeVector& pos, G4ThreeVector& mom, G4ThreeVector& pol, float& _t, float& _wavelength, int& _pmtid ) const
{
    assert( index < x.size() );

    pos.setX( x[index] );  
    pos.setY( y[index] );  
    pos.setZ( z[index] );  

    mom.setX( px[index] );  
    mom.setY( py[index] );  
    mom.setZ( pz[index] );  

    pol.setX( polx[index] );  
    pol.setY( poly[index] );  
    pol.setZ( polz[index] );  

    _t = t[index] ;
    _wavelength = wavelength[index] ;
    _pmtid = pmtid[index] ;

}
#endif

void ChromaPhotonList::GetPhoton(size_t index, 
                 float& _x,   float& _y, float& _z, 
                 float& _px,  float& _py, float& _pz, 
                 float& _polx,float& _poly ,float& _polz, 
                 float& _t, float& _wavelength, int& _pmtid ) const
{
    assert( index < x.size() );

    _x =  x[index] ;
    _y = y[index] ;  
    _z = z[index] ;  

    _px = px[index] ;  
    _py = py[index] ;  
    _pz = pz[index] ;  

    _polx = polx[index] ;  
    _poly = poly[index] ;  
    _polz = polz[index] ;  

    _t = t[index] ;
    _wavelength = wavelength[index] ;
    _pmtid = pmtid[index] ;

}


