DetSim DsG4Scintillation
==========================

Observations
-------------

#. `DsG4Scintillation` is applicable to OpticalPhotons (presumably for reemission handling), unlike `G4Scintillation`

   * hmm, maybe handling re-emission as separate process might be a more effcient way ?
   * how big does the re-emission tree get ?


Applicable to OP (Reemission)
--------------------------------

Applicable to OP and all::

    300 inline
    301 G4bool DsG4Scintillation::IsApplicable(const G4ParticleDefinition& aParticleType)
    302 {
    303         if (aParticleType.GetParticleName() == "opticalphoton"){
    304            return true;
    305         } else {
    306            return true;
    307         }
    308 }


Default is `doScintAndCeren`
------------------------------

* `DoBoth=true` means `doReemissionForOpticalPhotonsFromScintAndCeren`
* `DoBoth=false` means `doReemissionForOpticalPhotonsFromCeren`

::

    delta:src blyth$ grep DoBothProcess *.*
    DsG4Scintillation.h:        void SetDoBothProcess(bool tf = true) { doBothProcess = tf; }
    DsG4Scintillation.h:        bool GetDoBothProcess() { return doBothProcess; }
    DsPhysConsOptical.cc:    scint->SetDoBothProcess(m_doScintAndCeren);
    delta:src blyth$ 

Constants
----------

::

    246     if (aParticleName == "opticalphoton") {
    247       FastTimeConstant = "ReemissionFASTTIMECONSTANT";
    248       SlowTimeConstant = "ReemissionSLOWTIMECONSTANT";
    249       strYieldRatio = "ReemissionYIELDRATIO";
    250     }


Reemitted property set in TrackInfo
--------------------------------------

::

    685             G4Track* aSecondaryTrack =
    686                 new G4Track(aScintillationPhoton,aSecondaryTime,aSecondaryPosition);
    687 
    688             G4CompositeTrackInfo* comp=new G4CompositeTrackInfo();
    689             DsPhotonTrackInfo* trackinf=new DsPhotonTrackInfo();
    690             if ( flagReemission ){
    691                 if ( reemittedTI ) *trackinf = *reemittedTI;
    692                 trackinf->SetReemitted();
    693             }
    694             else if ( fApplyPreQE ) {
    695                 trackinf->SetMode(DsPhotonTrackInfo::kQEPreScale);
    696                 trackinf->SetQE(fPreQE);
    697             }
    698             comp->SetPhotonTrackInfo(trackinf);
    699             aSecondaryTrack->SetUserInformation(comp);
    ///
    /// parentage
    704 
    705             aSecondaryTrack->SetParentID(aTrack.GetTrackID());
    706 
    707             // add the secondary to the ParticleChange object
    708             aParticleChange.SetSecondaryWeightByProcess( true ); // recommended
    709             aParticleChange.AddSecondary(aSecondaryTrack);






