#include "CFG4_BODY.hh"

#include "CFG4_PUSH.hh"
#include "SteppingVerbose.hh"

#include "G4SteppingManager.hh"
#include "G4UnitsTable.hh"
#include "CFG4_POP.hh"


SteppingVerbose::SteppingVerbose()
 : G4SteppingVerbose()
{
}


SteppingVerbose::~SteppingVerbose()
{}


void SteppingVerbose::StepInfo()
{
  CopyState();

  verboseLevel = 5 ; 
  VerboseTrack(); 

  G4int prec = G4int(G4cout.precision(3));

  if( verboseLevel >= 1 ){
    if( verboseLevel >= 4 ) VerboseTrack();
    if( verboseLevel >= 3 ){
      G4cout << G4endl;
      G4cout << std::setw( 5) << "#Step#"     << " "
             << std::setw( 6) << "X"          << "    "
             << std::setw( 6) << "Y"          << "    "
             << std::setw( 6) << "Z"          << "    "
             << std::setw( 9) << "KineE"      << " "
             << std::setw( 9) << "dEStep"     << " "
             << std::setw(10) << "StepLeng"
             << std::setw(10) << "TrakLeng"
             << std::setw(10) << "Volume"    << "  "
             << std::setw(10) << "Process"   << G4endl;
    }

    G4cout << std::setw(5) << fTrack->GetCurrentStepNumber() << " "
    << std::setw(6) << G4BestUnit(fTrack->GetPosition().x(),"Length")
    << std::setw(6) << G4BestUnit(fTrack->GetPosition().y(),"Length")
    << std::setw(6) << G4BestUnit(fTrack->GetPosition().z(),"Length")
    << std::setw(6) << G4BestUnit(fTrack->GetKineticEnergy(),"Energy")
    << std::setw(6) << G4BestUnit(fStep->GetTotalEnergyDeposit(),"Energy")
    << std::setw(6) << G4BestUnit(fStep->GetStepLength(),"Length")
    << std::setw(6) << G4BestUnit(fTrack->GetTrackLength(),"Length")
    << "  ";

    // if( fStepStatus != fWorldBoundary){
    if( fTrack->GetNextVolume() != 0 ) {
      G4cout << std::setw(10) << fTrack->GetVolume()->GetName();
    } else {
      G4cout << std::setw(10) << "OutOfWorld";
    }

    if(fStep->GetPostStepPoint()->GetProcessDefinedStep() != 0){
      G4cout << "  "
             << std::setw(10)
             << fStep->GetPostStepPoint()->GetProcessDefinedStep()
                                         ->GetProcessName();
    } else {
      G4cout << "   UserLimit";
    }

    G4cout << G4endl;

    if( verboseLevel == 2 ){
      G4int tN2ndariesTot = fN2ndariesAtRestDoIt +
                            fN2ndariesAlongStepDoIt +
                            fN2ndariesPostStepDoIt;
      if(tN2ndariesTot>0){
        G4cout << "    :----- List of 2ndaries - "
               << "#SpawnInStep=" << std::setw(3) << tN2ndariesTot 
               << "(Rest="  << std::setw(2) << fN2ndariesAtRestDoIt
               << ",Along=" << std::setw(2) << fN2ndariesAlongStepDoIt
               << ",Post="  << std::setw(2) << fN2ndariesPostStepDoIt
               << "), "
               << "#SpawnTotal=" << std::setw(3) << (*fSecondary).size()
               << " ---------------"
               << G4endl;

        for(size_t lp1=(*fSecondary).size()-tN2ndariesTot;
                        lp1<(*fSecondary).size(); lp1++){
          G4cout << "    : "
                 << std::setw(6)
                 << G4BestUnit((*fSecondary)[lp1]->GetPosition().x(),"Length")
                 << std::setw(6)
                 << G4BestUnit((*fSecondary)[lp1]->GetPosition().y(),"Length")
                 << std::setw(6)
                 << G4BestUnit((*fSecondary)[lp1]->GetPosition().z(),"Length")
                 << std::setw(6)
                 << G4BestUnit((*fSecondary)[lp1]->GetKineticEnergy(),"Energy")
                 << std::setw(10)
                 << (*fSecondary)[lp1]->GetDefinition()->GetParticleName();
          G4cout << G4endl;
        }
 
        G4cout << "    :-----------------------------"
               << "----------------------------------"
               << "-- EndOf2ndaries Info ---------------"
               << G4endl;
      }
    }

  }
  G4cout.precision(prec);
}


void SteppingVerbose::TrackingStarted()
{

  CopyState();
  G4int prec = G4int(G4cout.precision(3));
  if( verboseLevel > 0 ){

    G4cout << std::setw( 5) << "Step#"      << " "
           << std::setw( 6) << "X"          << "    "
           << std::setw( 6) << "Y"          << "    "  
           << std::setw( 6) << "Z"          << "    "
           << std::setw( 9) << "KineE"      << " "
           << std::setw( 9) << "dEStep"     << " "
           << std::setw(10) << "StepLeng"
           << std::setw(10) << "TrakLeng"
           << std::setw(10) << "Volume"     << "  "
           << std::setw(10) << "Process"    << G4endl;

   G4cout << std::setw( 5) << fTrack->GetCurrentStepNumber() << " "
    << std::setw( 6) << G4BestUnit(fTrack->GetPosition().x(),"Length")
    << std::setw( 6) << G4BestUnit(fTrack->GetPosition().y(),"Length")
    << std::setw( 6) << G4BestUnit(fTrack->GetPosition().z(),"Length")
    << std::setw( 6) << G4BestUnit(fTrack->GetKineticEnergy(),"Energy")
    << std::setw( 6) << G4BestUnit(fStep->GetTotalEnergyDeposit(),"Energy")
    << std::setw( 6) << G4BestUnit(fStep->GetStepLength(),"Length")
    << std::setw( 6) << G4BestUnit(fTrack->GetTrackLength(),"Length")
    << "  ";

    if(fTrack->GetNextVolume()){
      G4cout << std::setw(10) << fTrack->GetVolume()->GetName();
    } else {
      G4cout << "OutOfWorld";
    }
    G4cout << "    initStep" << G4endl;
  }
  G4cout.precision(prec);
}

