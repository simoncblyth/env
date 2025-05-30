#include "CFG4_BODY.hh"
#include <cstring>

#include "G4Geantino.hh"
#include "G4ParticleTable.hh"
#include "G4ParticleDefinition.hh"

#include "G4SystemOfUnits.hh"
#include "G4PhysicalConstants.hh"

#include "CSource.hh"
#include "PLOG.hh"


CSource::part_prop_t::part_prop_t() 
{
  momentum_direction = G4ParticleMomentum(0,0,-1);
  energy = 1.*MeV;
  position = G4ThreeVector();
}





CSource::CSource(int verbosity)  
    :
    m_recorder(NULL),
	m_num(1),
	m_time(0.0),
	m_polarization(1.0,0.0,0.0),
    m_verbosityLevel(verbosity)
{
    init();
}


CSource::~CSource()
{
}  

void CSource::setRecorder(CRecorder* recorder)
{
   m_recorder = recorder ;  
}




void CSource::SetNumberOfParticles(G4int num) 
{
    m_num = num;
}
void CSource::SetParticleTime(G4double time) 
{
    m_time = time;
}
void CSource::SetParticlePolarization(G4ThreeVector polarization) 
{
    m_polarization = polarization ;
}
void CSource::SetParticlePosition(G4ThreeVector position) 
{
    part_prop_t& pp = m_pp.Get();
    pp.position = position ; 
}
void CSource::SetParticleMomentumDirection(G4ThreeVector direction) 
{
    part_prop_t& pp = m_pp.Get();
    pp.momentum_direction = direction  ; 
}
void CSource::SetParticleEnergy(G4double energy) 
{
    part_prop_t& pp = m_pp.Get();
    pp.energy = energy ; 
}



G4int CSource::GetNumberOfParticles() const 
{
    return m_num ;
}
G4ParticleDefinition* CSource::GetParticleDefinition() const 
{
    return m_definition;
}
G4double CSource::GetParticleTime() const 
{
    return m_time;
}

G4ThreeVector CSource::GetParticlePolarization() const 
{
    return m_polarization;
}
G4ThreeVector CSource::GetParticlePosition() const 
{
    return m_pp.Get().position;
}
G4ThreeVector CSource::GetParticleMomentumDirection() const 
{
    return m_pp.Get().momentum_direction;
}
G4double CSource::GetParticleEnergy() const 
{
    return m_pp.Get().energy;
}





void CSource::init()
{
	m_definition = G4Geantino::GeantinoDefinition();
}

void CSource::SetVerbosity(int vL) 
{
    G4AutoLock l(&m_mutex);
	m_verbosityLevel = vL;
}

void CSource::setParticle(const char* name)
{ 
	G4ParticleDefinition* definition = G4ParticleTable::GetParticleTable()->FindParticle(name);
    SetParticleDefinition(definition);
}

void CSource::SetParticleDefinition(G4ParticleDefinition* definition)
{ 
    m_definition = definition ; 
}





