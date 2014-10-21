#include "G4DAEChroma/G4DAEChroma.hh"

#include "G4DAEChroma/G4DAETransport.hh"
#include "G4DAEChroma/G4DAEGeometry.hh"
#include "G4DAEChroma/G4DAESensDet.hh"
#include "G4DAEChroma/G4DAETrojanSensDet.hh"

#include <iostream>


using namespace std ; 

G4DAEChroma* G4DAEChroma::fG4DAEChroma = 0;


G4DAEChroma* G4DAEChroma::GetG4DAEChroma()
{
  if(!fG4DAEChroma)
  {
     fG4DAEChroma = new G4DAEChroma;
  }
  return fG4DAEChroma;
}

G4DAEChroma* G4DAEChroma::GetG4DAEChromaIfExists()
{ 
  return fG4DAEChroma ;
}


G4DAEChroma::G4DAEChroma() :
    fTransport(0),
    fGeometry(0),
    fSensDet(0)
{ 
}

void G4DAEChroma::BeginOfRun( const G4Run* run )
{
    cout << "G4DAEChroma::BeginOfRun " << endl ;
}
void G4DAEChroma::EndOfRun(   const G4Run* run )
{
    cout << "G4DAEChroma::EndOfRun " << endl ;
}

void G4DAEChroma::Configure(const char* transport, const char* sensdet, const char* geometry)
{
    cout << "G4DAEChroma::Configure " << endl ;
    G4DAETransport* tra =  G4DAETransport::MakeTransport(transport);
    G4DAEGeometry*  geo =  G4DAEGeometry::MakeGeometry(geometry);
    G4DAETrojanSensDet* tsd = G4DAETrojanSensDet::MakeTrojanSensDet(sensdet, geo ); 

    this->SetSensDet( tsd );  
    this->SetGeometry( geo );  
    this->SetTransport( tra );
}





G4DAEChroma::~G4DAEChroma()
{
    if(fTransport) delete fTransport ;
    if(fGeometry)  delete fGeometry ;
    if(fSensDet)  delete fSensDet;

}

void G4DAEChroma::SetGeometry(G4DAEGeometry* geo){
   fGeometry = geo ; 
}
G4DAEGeometry* G4DAEChroma::GetGeometry(){
   return fGeometry ;
}


void G4DAEChroma::SetTransport(G4DAETransport* tra){
   fTransport = tra ; 
}
G4DAETransport* G4DAEChroma::GetTransport(){
   return fTransport ;
}

void G4DAEChroma::SetSensDet(G4DAESensDet* sd){
   fSensDet = sd ; 
}
G4DAESensDet* G4DAEChroma::GetSensDet(){
   return fSensDet ;
}




void G4DAEChroma::CollectPhoton(const G4Track* track)
{
   fTransport->CollectPhoton(track);
}


std::size_t G4DAEChroma::Propagate(G4int batch_id)
{
  std::size_t nhits = fTransport->Propagate(batch_id); 
  if(nhits > 0)
  { 
      fSensDet->CollectHits( fTransport->GetHits() );
  } 
  return nhits ; 
}





