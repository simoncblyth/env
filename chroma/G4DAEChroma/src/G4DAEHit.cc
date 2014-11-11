#include "G4DAEChroma/G4DAEHit.hh"
#include "G4DAEChroma/G4DAEPhotons.hh"  
#include "G4AffineTransform.hh"
#include <iostream>
using namespace std ;

void G4DAEHit::Init(G4DAEPhotons* photons, std::size_t index)
{
    // index is input, others are struct members that are hearby populated
    photons->GetPhoton( index, gpos, gdir, gpol, t, wavelength, pmtid );    
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

void G4DAEHit::Print()
{
    cout 
          << " pmtid "       << pmtid 
          << " t "     << t 
          << " wavelength " << wavelength 
          << " gpos "  << gpos 
          << " gdir "  << gdir 
          << " gpol "  << gpol 
          << " lpos "  << lpos 
          << " ldir "  << ldir 
          << " lpol "  << ldir 
          << endl ; 
}


void G4DAEHit::InitFake( std::size_t sensor_id, std::size_t track_id )
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

     pmtid = sensor_id ; 
     trackid = track_id  ;

}





