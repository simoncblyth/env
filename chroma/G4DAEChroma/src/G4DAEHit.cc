#include "G4DAEChroma/G4DAEHit.hh"
#include "Chroma/ChromaPhotonList.hh"  
#include <iostream>
using namespace std ;

void G4DAEHit::Init(ChromaPhotonList* cpl, std::size_t index)
{
    cpl->GetPhoton( index, gpos, gdir, gpol, t, wavelength, pmtid );    
}

void G4DAEHit::LocalTransform(G4AffineTransform& trans)
{ 
    lpos = trans.TransformPoint(gpos);
    lpol = trans.TransformAxis(gpol);
    lpol = lpol.unit();
    ldir = trans.TransformAxis(gdir);
    ldir = ldir.unit();
}

void G4DAEHit::Print()
{
    cout 
          << " volumeindex " << volumeindex 
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




