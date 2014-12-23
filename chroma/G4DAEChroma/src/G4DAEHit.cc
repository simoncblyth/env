#include "G4DAEChroma/G4DAEHit.hh"
#include "G4DAEChroma/G4DAEArrayHolder.hh"  
#include "G4DAEChroma/G4DAEPhotonList.hh"  
#include "G4DAEChroma/G4DAEPhoton.hh"  
#include "G4DAEChroma/G4DAECommon.hh"  

#include "G4AffineTransform.hh"
#include <iostream>
#include <cassert>
using namespace std ;


void G4DAEHit::Init(G4DAEArrayHolder* photons, std::size_t index)
{
    // index is input, others are struct members that are hearby populated

    G4DAEPhotonList* pl = new G4DAEPhotonList(photons);
    G4DAEPhoton::Get( pl, index, gpos, gdir, gpol, t, wavelength, pmtid );    

    // ensure initialize all elements of struct, otherwise get random bits
    weight = 1. ;
    photonid = index ;
    spare = 0 ;
}


void G4DAEHit::Serialize(float* data)
{
    data[_gpos_x] =  gpos.x() ;
    data[_gpos_y] =  gpos.y() ;
    data[_gpos_z] =  gpos.z() ;

    data[_gdir_x] =  gdir.x() ;
    data[_gdir_y] =  gdir.y() ;
    data[_gdir_z] =  gdir.z() ;

    data[_gpol_x] =  gpol.x() ;
    data[_gpol_y] =  gpol.y() ;
    data[_gpol_z] =  gpol.z() ;

    data[_lpos_x] =  lpos.x() ;
    data[_lpos_y] =  lpos.y() ;
    data[_lpos_z] =  lpos.z() ;

    data[_ldir_x] =  ldir.x() ;
    data[_ldir_y] =  ldir.y() ;
    data[_ldir_z] =  ldir.z() ;

    data[_lpol_x] =  lpol.x() ;
    data[_lpol_y] =  lpol.y() ;
    data[_lpol_z] =  lpol.z() ;

    data[_t]          =  t ;
    data[_wavelength] =  wavelength ;
    data[_weight]     =  weight ;

    uif_t uifd[3] ; 
    uifd[0].i = pmtid ;
    uifd[1].i = photonid ;
    uifd[2].i = spare     ;

    data[_pmtid]   =  uifd[0].f ;
    data[_photonid] =  uifd[1].f ;
    data[_spare]   =  uifd[2].f ;
}



void G4DAEHit::InitFake( std::size_t pmtid_, std::size_t photonid_ )
{
     gpos = G4ThreeVector();
     gdir = G4ThreeVector();
     gpol = G4ThreeVector();

     lpos = G4ThreeVector();
     ldir = G4ThreeVector();
     lpol = G4ThreeVector();

     t = 0. ; 
     wavelength = 0. ; 
     weight = 1. ;     

     pmtid = pmtid_ ; 
     photonid = photonid_  ;
     spare = 0 ;
}





void G4DAEHit::LocalTransform(G4AffineTransform* trans)
{ 
    if ( trans == NULL )
    {   
        //cout << "G4DAEHit::LocalTransform NULL transform " << endl ; 
        lpos = gpos ; 
        lpol = gpol ;
        ldir = gdir ; 
    }
    else
    {
        lpos = trans->TransformPoint(gpos);
        lpol = trans->TransformAxis(gpol);
        lpol = lpol.unit();
        ldir = trans->TransformAxis(gdir);
        ldir = ldir.unit();
    }
}

void G4DAEHit::Print(const char* msg) const
{
    cout  << msg 
          << " photonid "  << photonid 
          << " pmtid "      << pmtid 
          << " t "          << t 
          << " wavelength " << wavelength 
          << " gpos "  << gpos 
          << " gdir "  << gdir 
          << " gpol "  << gpol 
          << " lpos "  << lpos 
          << " ldir "  << ldir 
          << " lpol "  << ldir 
          << endl ; 
}


