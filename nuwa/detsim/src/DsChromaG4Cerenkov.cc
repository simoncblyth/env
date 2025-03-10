

#define G4DAECHROMA

#ifdef G4DAECHROMA
#include "G4DAEChroma/G4DAEChroma.hh"
#include "G4DAEChroma/G4DAECommon.hh"
static int bialkaliMaterialIndex = -1 ;
#endif


/**
 * \class DsChromaG4Cerenkov
 *
 * \brief A slightly modified version of G4Cerenkov
 *
 * It is modified to take a given weight to use to reduce the number
 * of opticalphotons that are produced.  They can then later be
 * up-weighted.
 *
 * The modification adds the weight, its accessors and adds
 * implementation to AlongStepDoIt().  We must copy-and-modify instead
 * of inherit because certain needed data members are private and so
 * we can not just override AlongStepDoIt() in our own subclass.
 *
 * This was taken from G4.9.1p1
 *
 * bv@bnl.gov Mon Feb  4 15:52:16 2008
 * Initial mod to support weighted opticalphotons.
 * The mods to dywCerenkov by Jianglai 09-06-2006 were used for guidance.
 *
 * Jul. 27, 2009  wangzhe
 *     ApplyWaterQe: apply all available QE when optical photons are created.
 *     This should be used with WaterCerenQeApplied of DsPmtSensDet.
 *     All modification are enclosed by "wangzhe" and "wz" for
 *     begin and end respectively.
 *     m_qeScale, etc. were copied to here from DsPmtSensDet.
 */

//
// ********************************************************************
// * License and Disclaimer                                           *
// *                                                                  *
// * The  Geant4 software  is  copyright of the Copyright Holders  of *
// * the Geant4 Collaboration.  It is provided  under  the terms  and *
// * conditions of the Geant4 Software License,  included in the file *
// * LICENSE and available at  http://cern.ch/geant4/license .  These *
// * include a list of copyright holders.                             *
// *                                                                  *
// * Neither the authors of this software system, nor their employing *
// * institutes,nor the agencies providing financial support for this *
// * work  make  any representation or  warranty, express or implied, *
// * regarding  this  software system or assume any liability for its *
// * use.  Please see the license in the file  LICENSE  and URL above *
// * for the full disclaimer and the limitation of liability.         *
// *                                                                  *
// * This  code  implementation is the result of  the  scientific and *
// * technical work of the GEANT4 collaboration.                      *
// * By using,  copying,  modifying or  distributing the software (or *
// * any work based  on the software)  you  agree  to acknowledge its *
// * use  in  resulting  scientific  publications,  and indicate your *
// * acceptance of all terms of the Geant4 Software license.          *
// ********************************************************************
//
//
// $Id: G4Cerenkov.cc,v 1.26 2008/11/14 20:16:51 gum Exp $
// GEANT4 tag $Name: geant4-09-02 $
//
////////////////////////////////////////////////////////////////////////
// Cerenkov Radiation Class Implementation
////////////////////////////////////////////////////////////////////////
//
// File:        G4Cerenkov.cc 
// Description: Discrete Process -- Generation of Cerenkov Photons
// Version:     2.1
// Created:     1996-02-21  
// Author:      Juliet Armstrong
// Updated:     2007-09-30 by Peter Gumplinger
//              > change inheritance to G4VDiscreteProcess
//              GetContinuousStepLimit -> GetMeanFreePath (StronglyForced)
//              AlongStepDoIt -> PostStepDoIt
//              2005-08-17 by Peter Gumplinger
//              > change variable name MeanNumPhotons -> MeanNumberOfPhotons
//              2005-07-28 by Peter Gumplinger
//              > add G4ProcessType to constructor
//              2001-09-17, migration of Materials to pure STL (mma) 
//              2000-11-12 by Peter Gumplinger
//              > add check on CerenkovAngleIntegrals->IsFilledVectorExist()
//              in method GetAverageNumberOfPhotons 
//              > and a test for MeanNumberOfPhotons <= 0.0 in DoIt
//              2000-09-18 by Peter Gumplinger
//              > change: aSecondaryPosition=x0+rand*aStep.GetDeltaPosition();
//                        aSecondaryTrack->SetTouchable(0);
//              1999-10-29 by Peter Gumplinger
//              > change: == into <= in GetContinuousStepLimit
//              1997-08-08 by Peter Gumplinger
//              > add protection against /0
//              > G4MaterialPropertiesTable; new physics/tracking scheme
//
// mail:        gum@triumf.ca
//
////////////////////////////////////////////////////////////////////////

#include "G4ios.hh"
#include "G4Poisson.hh"
#include "G4EmProcessSubType.hh"

#include "G4LossTableManager.hh"
#include "G4MaterialCutsCouple.hh"
#include "G4ParticleDefinition.hh"

#include "DsChromaG4Cerenkov.h"

#include "DsChromaPhotonTrackInfo.h"
#include "G4DataHelpers/G4CompositeTrackInfo.h"
using namespace std;

/////////////////////////
// Class Implementation  
/////////////////////////

        //////////////
        // Operators
        //////////////

// G4Cerenkov::operator=(const G4Cerenkov &right)
// {
// }

        /////////////////
        // Constructors
        /////////////////

DsChromaG4Cerenkov::DsChromaG4Cerenkov(const G4String& processName, G4ProcessType type)
           : G4VProcess(processName, type)
           , fApplyPreQE(false)
           , fPreQE(1.)
{
        G4cout << "DsChromaG4Cerenkov::DsChromaG4Cerenkov constructor" << G4endl;
        G4cout << "NOTE: this is now a G4VProcess!" << G4endl;
        G4cout << "Required change in UserPhysicsList: " << G4endl;
        G4cout << "change: pmanager->AddContinuousProcess(theCerenkovProcess);" << G4endl; // 
        G4cout << "to:     pmanager->AddProcess(theCerenkovProcess);" << G4endl;
        G4cout << "        pmanager->SetProcessOrdering(theCerenkovProcess,idxPostStep);" << G4endl;

        SetProcessSubType(fCerenkov);

        fTrackSecondariesFirst = false;
        fMaxBetaChange = 0.;
        fMaxPhotons = 0;
        fPhotonWeight = 1.0;    // Daya Bay mod, bv@bnl.gov

        thePhysicsTable = NULL;

        if (verboseLevel>0) {
            G4cout << GetProcessName() << " is created " << G4endl;
        }

        BuildThePhysicsTable();

        // wangzhe
        fApplyWaterQe = false;
        m_qeScale = 1.0/0.9;
        // wz
}

// G4Cerenkov::G4Cerenkov(const G4Cerenkov &right)
// {
// }

        ////////////////
        // Destructors
        ////////////////

DsChromaG4Cerenkov::~DsChromaG4Cerenkov() 
{
    if (thePhysicsTable != NULL) {
       thePhysicsTable->clearAndDestroy();
       delete thePhysicsTable;
    }
}

        ////////////
        // Methods
        ////////////

// PostStepDoIt
// -------------
//
G4VParticleChange*
DsChromaG4Cerenkov::PostStepDoIt(const G4Track& aTrack, const G4Step& aStep)

// This routine is called for each tracking Step of a charged particle
// in a radiator. A Poisson-distributed number of photons is generated
// according to the Cerenkov formula, distributed evenly along the track
// segment and uniformly azimuth w.r.t. the particle direction. The 
// parameters are then transformed into the Master Reference System, and 
// they are added to the particle change. 

{
	//////////////////////////////////////////////////////
	// Should we ensure that the material is dispersive?
	//////////////////////////////////////////////////////

#ifdef G4DAECHROMA
        G4DAEChroma* chroma = G4DAEChroma::GetG4DAEChroma();
        //
        // TODO: move config grabbing to filling a struct, 
        //  so dont have to repeat for every step
        //  difficult to do from ctor as chroma not yet initialized ?
        //

        size_t FLAG_G4CERENKOV_PSDI = chroma->FindFlag("G4CERENKOV_PSDI");

        size_t TASK_COLLECT_STEP    = chroma->FindTask("G4CERENKOV_COLLECT_STEP");
        size_t TASK_COLLECT_PHOTON  = chroma->FindTask("G4CERENKOV_COLLECT_PHOTON");
        size_t TASK_APPLY_WATER_QE  = chroma->FindTask("G4CERENKOV_APPLY_WATER_QE");
        size_t TASK_ADD_SECONDARY   = chroma->FindTask("G4CERENKOV_ADD_SECONDARY");
        size_t TASK_KILL_SECONDARY  = chroma->FindTask("G4CERENKOV_KILL_SECONDARY");

        chroma->Register(FLAG_G4CERENKOV_PSDI, 10000);
#endif
 
        aParticleChange.Initialize(aTrack);

        const G4DynamicParticle* aParticle = aTrack.GetDynamicParticle();
        const G4Material* aMaterial = aTrack.GetMaterial();

        G4StepPoint* pPreStepPoint  = aStep.GetPreStepPoint();
        G4StepPoint* pPostStepPoint = aStep.GetPostStepPoint();

        G4ThreeVector x0 = pPreStepPoint->GetPosition();
        G4ThreeVector p0 = aStep.GetDeltaPosition().unit();
        G4double t0 = pPreStepPoint->GetGlobalTime();

        G4MaterialPropertiesTable* aMaterialPropertiesTable =
                               aMaterial->GetMaterialPropertiesTable();
        if (!aMaterialPropertiesTable) return pParticleChange;

        const G4MaterialPropertyVector* Rindex = 
                aMaterialPropertiesTable->GetProperty("RINDEX"); 
        if (!Rindex) return pParticleChange;

        // particle charge
        const G4double charge = aParticle->GetDefinition()->GetPDGCharge();

        // particle beta
        const G4double beta = (pPreStepPoint ->GetBeta() +
                               pPostStepPoint->GetBeta())/2.;

        G4double MeanNumberOfPhotons = 
                 GetAverageNumberOfPhotons(charge,beta,aMaterial,Rindex);

        if (MeanNumberOfPhotons <= 0.0) {

                // return unchanged particle and no secondaries

                aParticleChange.SetNumberOfSecondaries(0);

                return pParticleChange;

        }

        G4double step_length;
        step_length = aStep.GetStepLength();

        MeanNumberOfPhotons = MeanNumberOfPhotons * step_length;
        G4bool ApplyWaterQE = fApplyWaterQe && aMaterial->GetName().contains("Water");

       // Reduce generated photons by given photon weight
       // Daya Bay mod, bv@bnl.gov
       if (verboseLevel>0) {
           G4cout << "DsChromaG4Cerenkov MeanNumberOfPhotons "<< MeanNumberOfPhotons 
                  << " before dividing by fPhotonWeight " << fPhotonWeight << G4endl;
       }
       MeanNumberOfPhotons/=fPhotonWeight;
       if (verboseLevel>0) {
           G4cout << "DsChromaG4Cerenkov MeanNumberOfPhotons "<< MeanNumberOfPhotons 
                  << " before multiplying by fPreQE " << fPreQE 
                  << " (only if fApplyPreQE=" << fApplyPreQE << " is set true " << G4endl;
       }
       if ( fApplyPreQE ) {
            // if WaterQE is applied, it's corrected by the fPreQE.
           MeanNumberOfPhotons *= fPreQE;
       }
       G4int NumPhotons = (G4int) G4Poisson(MeanNumberOfPhotons);
       if (verboseLevel>0) {
            G4cout << "DsChromaG4Cerenkov MeanNumberOfPhotons "<< MeanNumberOfPhotons
                   << " as mean of poission used to calculate NumPhotons " << NumPhotons
                   << G4endl;
       }
       if (NumPhotons <= 0) {
           // return unchanged particle and no secondaries  
            aParticleChange.SetNumberOfSecondaries(0);
           return pParticleChange;
       }

////////////////////////////////////////////////////////////////

       aParticleChange.SetNumberOfSecondaries(NumPhotons);

        if (fTrackSecondariesFirst) {
           if (aTrack.GetTrackStatus() == fAlive )
                   aParticleChange.ProposeTrackStatus(fSuspend);
        }

////////////////////////////////////////////////////////////////

       G4double Pmin = Rindex->GetMinPhotonEnergy();
       G4double Pmax = Rindex->GetMaxPhotonEnergy();
       G4double dp = Pmax - Pmin;

       G4double nMax = Rindex->GetMaxProperty();

       G4double BetaInverse = 1./beta;

       G4double maxCos = BetaInverse / nMax; 
       G4double maxSin2 = (1.0 - maxCos) * (1.0 + maxCos);

       const G4double beta1 = pPreStepPoint ->GetBeta();
       const G4double beta2 = pPostStepPoint->GetBeta();

       G4double MeanNumberOfPhotons1 =
                     GetAverageNumberOfPhotons(charge,beta1,aMaterial,Rindex);
       G4double MeanNumberOfPhotons2 =
                     GetAverageNumberOfPhotons(charge,beta2,aMaterial,Rindex);

#ifdef G4DAECHROMA
    // here for visibility from CerenkovPhoton collection
    size_t csid ;  
    G4int chromaMaterialIndex ; 
    if(TASK_COLLECT_STEP)
    {
        chroma->Start(TASK_COLLECT_STEP);

        if(bialkaliMaterialIndex == -1 )
        {
              G4Material* bialkali = G4Material::GetMaterial("/dd/Materials/Bialkali");
              bialkaliMaterialIndex = bialkali ? bialkali->GetIndex() : -2 ; 
        }
        assert(bialkaliMaterialIndex > -1 );

        // serialize DsG4Cerenkov::PostStepDoIt stack, just before the photon loop
        G4DAECerenkovStepList* csl = chroma->GetCerenkovStepList();
        int* g2c = chroma->GetMaterialLookup();

        const G4ParticleDefinition* definition = aParticle->GetDefinition(); 
        G4ThreeVector deltaPosition = aStep.GetDeltaPosition();
        G4double weight = fPhotonWeight*aTrack.GetWeight();
        G4int materialIndex = aMaterial->GetIndex();
        assert(materialIndex > -1 );

        // this relates Geant4 materialIndex to the chroma equivalent
        chromaMaterialIndex = g2c[materialIndex] ;
        assert(chromaMaterialIndex > -1 );

        G4int chromaBialkaliIndex = g2c[bialkaliMaterialIndex] ;
        G4String materialName = aMaterial->GetName();

        csid = 1 + csl->GetCount() ;  // 1-based

        /*
        cout << "G4DAECerenkovStep " 
             << " csid " << csid 
             << " materialIndex " << materialIndex
             << " chromaMaterialIndex " << chromaMaterialIndex
             << " materialName " << materialName
             << endl ;
        */ 

        // shoving ints into float bits
        uif_t uifa[4] ;
        uifa[0].i = -csid ;     //   negated 1-based index signalling Cerenkov (as opposed to Scintillation), acted upon in generate.cu
        uifa[1].i = aTrack.GetTrackID() ;
        uifa[2].i = chromaMaterialIndex ; 
        uifa[3].i = NumPhotons ;

        uif_t uifb[4] ;
        uifb[0].i = definition->GetPDGEncoding();
        uifb[1].i = chromaBialkaliIndex ;  
        uifb[2].i = 0 ;
        uifb[3].i = 0 ;


        // directly fills next item of G4DAEArray using (n,4,6) structure [float4 quads efficient on GPU]
        float* cs = csl->GetNextPointer();     

        cs[G4DAECerenkovStep::_Id]         =  uifa[0].f ; // 0
        cs[G4DAECerenkovStep::_ParentID]   =  uifa[1].f ;
        cs[G4DAECerenkovStep::_Material]   =  uifa[2].f ; 
        cs[G4DAECerenkovStep::_NumPhotons] =  uifa[3].f ;

        cs[G4DAECerenkovStep::_x0_x] =  x0.x() ;         // 1
        cs[G4DAECerenkovStep::_x0_y] =  x0.y() ;
        cs[G4DAECerenkovStep::_x0_z] =  x0.z() ;
        cs[G4DAECerenkovStep::_t0]   =  t0 ;

        cs[G4DAECerenkovStep::_DeltaPosition_x] =  deltaPosition.x() ;   // 2
        cs[G4DAECerenkovStep::_DeltaPosition_y] =  deltaPosition.y() ;
        cs[G4DAECerenkovStep::_DeltaPosition_z] =  deltaPosition.z() ;
        cs[G4DAECerenkovStep::_step_length] =  step_length ;
 
        cs[G4DAECerenkovStep::_code] = uifb[0].f ;    // 3
        cs[G4DAECerenkovStep::_charge] = charge ;
        cs[G4DAECerenkovStep::_weight] = weight ;
        cs[G4DAECerenkovStep::_MeanVelocity] = ((pPreStepPoint->GetVelocity()+pPostStepPoint->GetVelocity())/2.);

        cs[G4DAECerenkovStep::_BetaInverse] = BetaInverse ;   // 4
        cs[G4DAECerenkovStep::_Pmin] = Pmin ;
        cs[G4DAECerenkovStep::_Pmax] = Pmax ;
        cs[G4DAECerenkovStep::_maxCos] = maxCos ;

        cs[G4DAECerenkovStep::_maxSin2] = maxSin2 ;     // 5
        cs[G4DAECerenkovStep::_MeanNumberOfPhotons1] = MeanNumberOfPhotons1 ;
        cs[G4DAECerenkovStep::_MeanNumberOfPhotons2] = MeanNumberOfPhotons2 ;
        cs[G4DAECerenkovStep::_BialkaliMaterialIndex] = uifb[1].f ; 

        chroma->Stop(TASK_COLLECT_STEP);
    }
#endif

	for (G4int i = 0; i < NumPhotons; i++) {
	  // Determine photon energy
	  G4double rand=0;
	  G4double sampledEnergy=0, sampledRI=0; 
	  G4double cosTheta=0, sin2Theta=0;
	  
	  // sample an energy
	  do {
	    rand = G4UniformRand();	
	    sampledEnergy = Pmin + rand * dp; 
	    sampledRI = Rindex->GetProperty(sampledEnergy);
	    cosTheta = BetaInverse / sampledRI;  
	    
	    sin2Theta = (1.0 - cosTheta)*(1.0 + cosTheta);
	    rand = G4UniformRand();	
	    
	  } while (rand*maxSin2 > sin2Theta);
	 

	  // wangzhe
	  // kill a optical photon according to the QE(energy) probability function
	  G4double qe=1.;
	  if ( ApplyWaterQE ) 
      {
	      G4double uni;
	      // 0.6: For now, hard code "extra" decrease in efficiency for water shield PMTs to match G4dyb.
	      // m_qeScale: 1.0/0.9
	      G4double effqe=qe=0.6*m_qeScale*GetPoolPmtQe(sampledEnergy);
	      if ( fApplyPreQE ) 
          {
	          // take into account preapplied maximal QE 
	          effqe/=fPreQE;
	          if ( effqe>1. ) G4cerr<<"WaterPMT efficiency>1. This means that used CerenPhotonScaleWeight is too big."<<G4endl;
	      }
	      uni = G4UniformRand();
	      //G4cout <<"qe= "<<qe<<" uni= "<<uni<<" energy= "<<sampledEnergy/CLHEP::eV<<" eV, "
	      //	     <<"raw QE= "<<GetPoolPmtQe(sampledEnergy)<<G4endl;
	    
	      if ( uni >= effqe ) 
          {
#ifdef G4DAECHROMA
        if(TASK_APPLY_WATER_QE)
        {
	         continue;
        }
#endif
	      }
	  }
	  // wz
	  
	  // Generate random position of photon on cone surface 
	  // defined by Theta 
	  rand = G4UniformRand();
	  
	  G4double phi = twopi*rand;
	  G4double sinPhi = sin(phi);
	  G4double cosPhi = cos(phi);
	  
	  // calculate x,y, and z components of photon energy
	  // (in coord system with primary particle direction 
	  //  aligned with the z axis)
	  
	  G4double sinTheta = sqrt(sin2Theta); 
	  G4double px = sinTheta*cosPhi;
	  G4double py = sinTheta*sinPhi;
	  G4double pz = cosTheta;
	  
	  // Create photon momentum direction vector 
	  // The momentum direction is still with respect
	  // to the coordinate system where the primary
	  // particle direction is aligned with the z axis  
	  
	  G4ParticleMomentum photonMomentum(px, py, pz);
	  
	  // Rotate momentum direction back to global reference
	  // system 
	  
	  photonMomentum.rotateUz(p0);
	  
	  // Determine polarization of new photon 
	  
	  G4double sx = cosTheta*cosPhi;
	  G4double sy = cosTheta*sinPhi; 
	  G4double sz = -sinTheta;
	  
	  G4ThreeVector photonPolarization(sx, sy, sz);
	  
	  // Rotate back to original coord system 
	  
	  photonPolarization.rotateUz(p0);
	  
	  // Generate a new photon:
	  
	  G4DynamicParticle* aCerenkovPhoton =
	    new G4DynamicParticle(G4OpticalPhoton::OpticalPhoton(), 
				  photonMomentum);
	  aCerenkovPhoton->SetPolarization
	    (photonPolarization.x(),
	     photonPolarization.y(),
	     photonPolarization.z());
	  
	  aCerenkovPhoton->SetKineticEnergy(sampledEnergy);
	  
	  // Generate new G4Track object:
	  
	  G4double delta, NumberOfPhotons, N;
	  
	  do {
	    rand = G4UniformRand();
	    delta = rand * aStep.GetStepLength();
	    NumberOfPhotons = MeanNumberOfPhotons1 - delta *
	      (MeanNumberOfPhotons1-MeanNumberOfPhotons2)/
	      aStep.GetStepLength();
	    N = G4UniformRand() *
	      std::max(MeanNumberOfPhotons1,MeanNumberOfPhotons2);
	  } while (N > NumberOfPhotons);
	  
	  G4double deltaTime = delta /
	    ((pPreStepPoint->GetVelocity()+
	      pPostStepPoint->GetVelocity())/2.);
	  
	  G4double aSecondaryTime = t0 + deltaTime;
	  
	  G4ThreeVector aSecondaryPosition =
	    x0 + rand * aStep.GetDeltaPosition();
	  
	  G4Track* aSecondaryTrack = 
	    new G4Track(aCerenkovPhoton,aSecondaryTime,aSecondaryPosition);
	  
	  // set user track info
	  G4CompositeTrackInfo* comp=new G4CompositeTrackInfo();
	  DsChromaPhotonTrackInfo* trackinf=new DsChromaPhotonTrackInfo();
	  if ( ApplyWaterQE ) 
      {
	      trackinf->SetMode(DsChromaPhotonTrackInfo::kQEWater);
	      trackinf->SetQE(qe);
	  }
	  else if ( fApplyPreQE ) 
      {
	      trackinf->SetMode(DsChromaPhotonTrackInfo::kQEPreScale);
	      trackinf->SetQE(fPreQE);
	  }
	  comp->SetPhotonTrackInfo(trackinf);
	  aSecondaryTrack->SetUserInformation(comp);
	  
	  aSecondaryTrack->SetTouchableHandle(
					      aStep.GetPreStepPoint()->GetTouchableHandle());
	  
	  aSecondaryTrack->SetParentID(aTrack.GetTrackID());
	  

#ifdef G4DAECHROMA
    {
        if(TASK_ADD_SECONDARY)
        {
            chroma->Start(TASK_ADD_SECONDARY);

            aParticleChange.AddSecondary(aSecondaryTrack);

            chroma->Stop(TASK_ADD_SECONDARY);
        }
    }
#endif
	  
	  // Daya Bay mods, bv@bnl.gov
	  aSecondaryTrack->SetWeight(fPhotonWeight*aTrack.GetWeight());
	  aParticleChange.SetSecondaryWeightByProcess(true);
	  if (verboseLevel>0) {
	    G4cout << "DsChromaG4Cerenkov  aSecondaryTrack->SetWeight( fPhotonWeight="<<fPhotonWeight<<" * aTrack.GetWeight()= " << aTrack.GetWeight() 
		   << ") aSecondaryTrack->GetWeight() " << aSecondaryTrack->GetWeight() << G4endl;
	  }


#ifdef G4DAECHROMA
        if(TASK_COLLECT_PHOTON)
        {
            chroma->Start(TASK_COLLECT_PHOTON);

            G4DAECerenkovPhotonList* cpl = chroma->GetCerenkovPhotonList();
            //size_t cpid = 1 + cpl->GetCount() ;  // 1-based 
            float* cp = cpl->GetNextPointer();     

            float wavelength = (h_Planck * c_light / sampledEnergy) / nanometer ;

            cp[G4DAECerenkovPhoton::_post_x] = aSecondaryPosition.x()/mm ;
            cp[G4DAECerenkovPhoton::_post_y] = aSecondaryPosition.y()/mm ;
            cp[G4DAECerenkovPhoton::_post_z] = aSecondaryPosition.z()/mm ;
            cp[G4DAECerenkovPhoton::_post_w] = aSecondaryTime/ns ;

            cp[G4DAECerenkovPhoton::_dirw_x] = photonMomentum.x();
            cp[G4DAECerenkovPhoton::_dirw_y] = photonMomentum.y() ;
            cp[G4DAECerenkovPhoton::_dirw_z] = photonMomentum.z() ;
            cp[G4DAECerenkovPhoton::_dirw_w] = wavelength ; 

            cp[G4DAECerenkovPhoton::_polw_x] = photonPolarization.x();
            cp[G4DAECerenkovPhoton::_polw_y] = photonPolarization.y() ;
            cp[G4DAECerenkovPhoton::_polw_z] = photonPolarization.z() ;
            cp[G4DAECerenkovPhoton::_polw_w] = aSecondaryTrack->GetWeight() ; 

            uif_t uifd[4] ; 
            //uifd[0].i = cpid ;  // 1-based photon index 
            uifd[0].i = chromaMaterialIndex ;  // record material with photon  
            uifd[1].i = csid ;  // 1-based cerenkov step id
            uifd[2].u = 0 ;     // flags
            uifd[3].i = -1  ;   // pmtid

            cp[G4DAECerenkovPhoton::_flag_x] =  uifd[0].f ;
            cp[G4DAECerenkovPhoton::_flag_y] =  uifd[1].f ;
            cp[G4DAECerenkovPhoton::_flag_z] =  uifd[2].f ;
            cp[G4DAECerenkovPhoton::_flag_w] =  uifd[3].f ;

            chroma->Stop(TASK_COLLECT_PHOTON);
       } 
#endif


	}  // over NumPhotons



#ifdef G4DAECHROMA
    {
        if(TASK_KILL_SECONDARY)
        {
            chroma->Start(TASK_KILL_SECONDARY);

            if (verboseLevel > 0) 
            G4cout << "DsChromaG4Cerenkov::PostStepDoIt FLAG_G4CERENKOV_KILL_SECONDARY " << aParticleChange.GetNumberOfSecondaries() << " G4 cerenkov secondaries " << G4endl ;  
            aParticleChange.SetNumberOfSecondaries(0);

            chroma->Stop(TASK_KILL_SECONDARY);
            return pParticleChange;  // huh "a" "p" pattern used above 
        }
        else 
        {
            if (verboseLevel > 0) 
            G4cout << "DsChromaG4Cerenkov::PostStepDoIt proceed with " << aParticleChange.GetNumberOfSecondaries() << " G4 cerenkov secondaries " << G4endl ;  
        }
    }
#endif 
        return pParticleChange;
}

// BuildThePhysicsTable for the Cerenkov process
// ---------------------------------------------
//

void DsChromaG4Cerenkov::BuildThePhysicsTable()
{
	if (thePhysicsTable) return;

	const G4MaterialTable* theMaterialTable=
	 		       G4Material::GetMaterialTable();
	G4int numOfMaterials = G4Material::GetNumberOfMaterials();

	// create new physics table
	
	thePhysicsTable = new G4PhysicsTable(numOfMaterials);

	// loop for materials

        //G4cerr << "Building physics table with " << numOfMaterials << " materials" << G4endl;

	for (G4int i=0 ; i < numOfMaterials; i++)
	{
		G4PhysicsOrderedFreeVector* aPhysicsOrderedFreeVector =
					new G4PhysicsOrderedFreeVector();

		// Retrieve vector of refraction indices for the material
		// from the material's optical properties table 

		G4Material* aMaterial = (*theMaterialTable)[i];

		G4MaterialPropertiesTable* aMaterialPropertiesTable =
				aMaterial->GetMaterialPropertiesTable();

		if (aMaterialPropertiesTable) {

		   G4MaterialPropertyVector* theRefractionIndexVector = 
		    	   aMaterialPropertiesTable->GetProperty("RINDEX");

		   if (theRefractionIndexVector) {
		
		      // Retrieve the first refraction index in vector
		      // of (photon energy, refraction index) pairs 

		      theRefractionIndexVector->ResetIterator();
		      ++(*theRefractionIndexVector);	// advance to 1st entry 

		      G4double currentRI = theRefractionIndexVector->
		  			   GetProperty();

		      if (currentRI > 1.0) {

			 // Create first (photon energy, Cerenkov Integral)
			 // pair  

			 G4double currentPM = theRefractionIndexVector->
			 			 GetPhotonEnergy();
			 G4double currentCAI = 0.0;

			 aPhysicsOrderedFreeVector->
			 	 InsertValues(currentPM , currentCAI);

			 // Set previous values to current ones prior to loop

			 G4double prevPM  = currentPM;
			 G4double prevCAI = currentCAI;
                	 G4double prevRI  = currentRI;

			 // loop over all (photon energy, refraction index)
			 // pairs stored for this material  

			 while(++(*theRefractionIndexVector))
			 {
				currentRI=theRefractionIndexVector->	
						GetProperty();

				currentPM = theRefractionIndexVector->
						GetPhotonEnergy();

				currentCAI = 0.5*(1.0/(prevRI*prevRI) +
					          1.0/(currentRI*currentRI));

				currentCAI = prevCAI + 
					     (currentPM - prevPM) * currentCAI;

				aPhysicsOrderedFreeVector->
				    InsertValues(currentPM, currentCAI);

				prevPM  = currentPM;
				prevCAI = currentCAI;
				prevRI  = currentRI;
			 }

		      }
		   }
		}

	// The Cerenkov integral for a given material
	// will be inserted in thePhysicsTable
	// according to the position of the material in
	// the material table. 

	thePhysicsTable->insertAt(i,aPhysicsOrderedFreeVector); 

	}
}

// GetMeanFreePath
// ---------------
//

G4double DsChromaG4Cerenkov::GetMeanFreePath(const G4Track&,
                                           G4double,
                                           G4ForceCondition*)
{
        return 1.;
}

G4double DsChromaG4Cerenkov::PostStepGetPhysicalInteractionLength(
                                           const G4Track& aTrack,
                                           G4double,
                                           G4ForceCondition* condition)
{
        *condition = NotForced;
        G4double StepLimit = DBL_MAX;

        const G4DynamicParticle* aParticle = aTrack.GetDynamicParticle();
        const G4Material* aMaterial = aTrack.GetMaterial();
        const G4MaterialCutsCouple* couple = aTrack.GetMaterialCutsCouple();

        const G4double kineticEnergy = aParticle->GetKineticEnergy();
        const G4ParticleDefinition* particleType = aParticle->GetDefinition();
        const G4double mass = particleType->GetPDGMass();

        // particle beta
        const G4double beta = aParticle->GetTotalMomentum() /
                              aParticle->GetTotalEnergy();
        // particle gamma
        const G4double gamma = 1./std::sqrt(1.-beta*beta);

        G4MaterialPropertiesTable* aMaterialPropertiesTable =
                            aMaterial->GetMaterialPropertiesTable();

        const G4MaterialPropertyVector* Rindex = NULL;

        if (aMaterialPropertiesTable)
                     Rindex = aMaterialPropertiesTable->GetProperty("RINDEX");

        G4double nMax;
        if (Rindex) {
           nMax = Rindex->GetMaxProperty();
        } else {
           return StepLimit;
        }

        G4double BetaMin = 1./nMax;
        if ( BetaMin >= 1. ) return StepLimit;

        G4double GammaMin = 1./std::sqrt(1.-BetaMin*BetaMin);

        if (gamma < GammaMin ) return StepLimit;

        G4double kinEmin = mass*(GammaMin-1.);

        G4double RangeMin = G4LossTableManager::Instance()->
                                                   GetRange(particleType,
                                                            kinEmin,
                                                            couple);
        G4double Range    = G4LossTableManager::Instance()->
                                                   GetRange(particleType,
                                                            kineticEnergy,
                                                            couple);

        G4double Step = Range - RangeMin;
        if (Step < 1.*um ) return StepLimit;

        if (Step > 0. && Step < StepLimit) StepLimit = Step; 

        // If user has defined an average maximum number of photons to
        // be generated in a Step, then calculate the Step length for
        // that number of photons. 
 
        if (fMaxPhotons > 0) {

           // particle charge
           const G4double charge = aParticle->
                                   GetDefinition()->GetPDGCharge();

	   G4double MeanNumberOfPhotons = 
                    GetAverageNumberOfPhotons(charge,beta,aMaterial,Rindex);

           G4double Step = 0.;
           if (MeanNumberOfPhotons > 0.0) Step = fMaxPhotons /
                                                 MeanNumberOfPhotons;

           if (Step > 0. && Step < StepLimit) StepLimit = Step;
        }

        // If user has defined an maximum allowed change in beta per step
        if (fMaxBetaChange > 0.) {

           G4double dedx = G4LossTableManager::Instance()->
                                                   GetDEDX(particleType,
                                                           kineticEnergy,
                                                           couple);

           G4double deltaGamma = gamma - 
                                 1./std::sqrt(1.-beta*beta*
                                                 (1.-fMaxBetaChange)*
                                                 (1.-fMaxBetaChange));

           G4double Step = mass * deltaGamma / dedx;

           if (Step > 0. && Step < StepLimit) StepLimit = Step;

        }

        *condition = StronglyForced;
        return StepLimit;
}

// GetAverageNumberOfPhotons
// -------------------------
// This routine computes the number of Cerenkov photons produced per
// GEANT-unit (millimeter) in the current medium. 
//             ^^^^^^^^^^

G4double 
DsChromaG4Cerenkov::GetAverageNumberOfPhotons(const G4double charge,
                              const G4double beta, 
			      const G4Material* aMaterial,
			      const G4MaterialPropertyVector* Rindex) const
{
	const G4double Rfact = 369.81/(eV * cm);

        if(beta <= 0.0)return 0.0;

        G4double BetaInverse = 1./beta;

	// Vectors used in computation of Cerenkov Angle Integral:
	// 	- Refraction Indices for the current material
	//	- new G4PhysicsOrderedFreeVector allocated to hold CAI's
 
        //G4cerr << "In Material getting index: " << aMaterial->GetName() << G4endl;
	G4int materialIndex = aMaterial->GetIndex();
        //G4cerr << "\tindex=" << materialIndex << G4endl;

	// Retrieve the Cerenkov Angle Integrals for this material  
    // G4PhysicsVector* pv = (*thePhysicsTable)(materialIndex);
	G4PhysicsOrderedFreeVector* CerenkovAngleIntegrals =
	(G4PhysicsOrderedFreeVector*)((*thePhysicsTable)(materialIndex));

        if(!(CerenkovAngleIntegrals->IsFilledVectorExist()))return 0.0;

	// Min and Max photon energies 
	G4double Pmin = Rindex->GetMinPhotonEnergy();
	G4double Pmax = Rindex->GetMaxPhotonEnergy();

	// Min and Max Refraction Indices 
	G4double nMin = Rindex->GetMinProperty();	
	G4double nMax = Rindex->GetMaxProperty();

	// Max Cerenkov Angle Integral 
	G4double CAImax = CerenkovAngleIntegrals->GetMaxValue();

	G4double dp=0, ge=0;

	// If n(Pmax) < 1/Beta -- no photons generated 

	if (nMax < BetaInverse) {
		dp = 0;
		ge = 0;
	} 

	// otherwise if n(Pmin) >= 1/Beta -- photons generated  

	else if (nMin > BetaInverse) {
		dp = Pmax - Pmin;	
		ge = CAImax; 
	} 

	// If n(Pmin) < 1/Beta, and n(Pmax) >= 1/Beta, then
	// we need to find a P such that the value of n(P) == 1/Beta.
	// Interpolation is performed by the GetPhotonEnergy() and
	// GetProperty() methods of the G4MaterialPropertiesTable and
	// the GetValue() method of G4PhysicsVector.  

	else {
		Pmin = Rindex->GetPhotonEnergy(BetaInverse);
		dp = Pmax - Pmin;

		// need boolean for current implementation of G4PhysicsVector
		// ==> being phased out
		G4bool isOutRange;
		G4double CAImin = CerenkovAngleIntegrals->
                                  GetValue(Pmin, isOutRange);
		ge = CAImax - CAImin;

		if (verboseLevel>0) {
			G4cout << "CAImin = " << CAImin << G4endl;
			G4cout << "ge = " << ge << G4endl;
		}
	}
	
	// Calculate number of photons 
	G4double NumPhotons = Rfact * charge/eplus * charge/eplus *
                                 (dp - ge * BetaInverse*BetaInverse);

	return NumPhotons;		
}


// wangzhe
// get the raw pmt photocathode QE
G4double DsChromaG4Cerenkov::GetPoolPmtQe(G4double energy) const
{
  static bool first = true;
  static G4Material* bialkali = 0;
  if(first) {
    bialkali = G4Material::GetMaterial("/dd/Materials/Bialkali");
    if( bialkali ==0 ) {
      G4cout<<"Error: DsChromaG4Cerenkov::Can't find material bialkali."<<G4endl;
    }
    first = false;
  }
  
  G4MaterialPropertyVector* qevec = bialkali->GetMaterialPropertiesTable()->GetProperty("EFFICIENCY");
  return qevec->GetProperty(energy);

}
// wz
