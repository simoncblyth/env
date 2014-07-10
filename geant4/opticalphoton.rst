Geant4 Optical Photon
======================

* :google:`optical photon processes`

* http://geant4.slac.stanford.edu/UsersWorkshop/PDF/Peter/OpticalPhoton.pdf
* (ExampleN06 at /examples/novice/N06)

Production of OP (/processes/electromagnetic/xrays)
----------------------------------------------------

* Cerenkov Process 
* Scintillation Process 
* Transition Radiation 

Propagation of OP (/processes/optical)
----------------------------------------

* Refraction and Reflection at medium boundaries 
* Bulk Absorption 
* Rayleigh scattering 


Hypernews on opticalphotons
-----------------------------

* http://hypernews.slac.stanford.edu/HyperNews/geant4/get/opticalphotons.html

Min step size for optical photons
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* http://hypernews.slac.stanford.edu/HyperNews/geant4/get/opticalphotons/505/1.html

::

    > I'm wondering whether there is a way to set the minimum step size
    > for produced photons to make tracking less accurate, but also less CPU
    > consuming? I've searched forum, but found only examples of setting
    > maximum step size, which is not the case. If such a possibility exists,
    > I'd be grateful for every hint.

    The steps optical photons take during the simulation are already the
    minimum step size they can take. Optical photon steps are only limited by
    geometry and discrete processes. The geometry has to be interrogated to find
    the next volume boundary - I am not aware of any way to make this 'less
    accurate' but faster.

    If you find the CPU consum prohibitive you may bias the number of optical
    photons you track. For example, you can remove every other photon in your
    UserStackingAction. This reduces the statistical accuracy but you may still get
    a result that is statistical significant for your setup.

    Best regards, Peter




Geant4 OP processes
--------------------

::

    blyth@cms01 include]$ ll
    total 116
    -rw-r--r--  1 blyth blyth  2780 Mar 16  2009 G4WLSTimeGeneratorProfileExponential.hh
    -rw-r--r--  1 blyth blyth  2764 Mar 16  2009 G4WLSTimeGeneratorProfileDelta.hh
    -rw-r--r--  1 blyth blyth  2779 Mar 16  2009 G4VWLSTimeGeneratorProfile.hh
    -rw-r--r--  1 blyth blyth  5221 Mar 16  2009 G4OpWLS.hh
    -rw-r--r--  1 blyth blyth  5824 Mar 16  2009 G4OpRayleigh.hh
    -rw-r--r--  1 blyth blyth  2255 Mar 16  2009 G4OpProcessSubType.hh
    -rw-r--r--  1 blyth blyth 10395 Mar 16  2009 G4OpBoundaryProcess.hh.orig
    -rw-r--r--  1 blyth blyth  4597 Mar 16  2009 G4OpAbsorption.hh
    drwxr-xr-x  4 blyth blyth  4096 Mar 16  2009 ..
    -rw-r--r--  1 blyth blyth 11031 Feb 16  2011 G4OpBoundaryProcess.hh
    drwxr-xr-x  2 blyth blyth  4096 Feb 16  2011 .


    [blyth@cms01 include]$ grep public\ G4VDiscreteProcess *.hh
    G4OpAbsorption.hh:class G4OpAbsorption : public G4VDiscreteProcess 
    G4OpBoundaryProcess.hh:class G4OpBoundaryProcess : public G4VDiscreteProcess
    G4OpRayleigh.hh:class G4OpRayleigh : public G4VDiscreteProcess 
    G4OpWLS.hh:class G4OpWLS : public G4VDiscreteProcess 

::

    [blyth@cms01 include]$ grep processName *.hh
    G4OpAbsorption.hh:        G4OpAbsorption(const G4String& processName = "OpAbsorption",
    G4OpBoundaryProcess.hh:        G4OpBoundaryProcess(const G4String& processName = "OpBoundary",
    G4OpRayleigh.hh:        G4OpRayleigh(const G4String& processName = "OpRayleigh",
    G4OpWLS.hh:  G4OpWLS(const G4String& processName = "OpWLS",


`g4optical/include/G4OpProcessSubType.hh`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    enum G4OpProcessSubType 
    { 
      fOpAbsorption = 31,
      fOpBoundary = 32,
      fOpRayleigh = 33,
      fOpWLS = 34
    };



G4OpBoundaryProcess.hh patch
-------------------------------

::
    [blyth@cms01 include]$ pwd
    /data/env/local/dyb/trunk/external/build/LCG/geant4.9.2.p01/source/processes/optical/include
    [blyth@cms01 include]$ diff G4OpBoundaryProcess.hh.orig G4OpBoundaryProcess.hh
    302,305c302,318
    < 
    <           NewMomentum = G4LambertianRand(theGlobalNormal);
    <           theFacetNormal = (NewMomentum - OldMomentum).unit();
    < 
    ---
    >         // wangzhe
    >         // Original:
    >           //NewMomentum = G4LambertianRand(theGlobalNormal);
    >           //theFacetNormal = (NewMomentum - OldMomentum).unit();
    >         
    >         // Temp Fix:
    >         if(theGlobalNormal.mag()==0) {
    >             std::cout<<"Error. Zero caught. A normal vector with mag be 0. May trigger a infinite loop later."<<std::endl;
    >             std::cout<<"A temporary solution: Photon is forced to go back along its original path."<<std::endl;
    >             std::cout<<"Test from MDC09a tells the effect of this bug is tiny."<<std::endl;
    >           G4ThreeVector myVec(0,0,0);
    >           theFacetNormal = (myVec - OldMomentum).unit();
    >         } else {
    >           NewMomentum = G4LambertianRand(theGlobalNormal);
    >           theFacetNormal = (NewMomentum - OldMomentum).unit();
    >         }
    >         // wz




DayaBay DetSim specializations
-------------------------------

::

    [blyth@cms01 src]$ grep public\ G4VDiscreteProcess *.h
    DsG4OpBoundaryProcess.h:class DsG4OpBoundaryProcess : public G4VDiscreteProcess
    DsG4OpRayleigh.h:class DsG4OpRayleigh : public G4VDiscreteProcess 
    [blyth@cms01 src]$ pwd
    /data/env/local/dyb/trunk/NuWa-trunk/dybgaudi/Simulation/DetSim/src



:dybsvn:`source:dybgaudi/trunk/Simulation/DetSim/src/DsG4OpBoundaryProcess.h`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Same difference as noted above, plus name changes::

    [blyth@cms01 src]$ diff ~/g4optical/include/G4OpBoundaryProcess.hh DsG4OpBoundaryProcess.h


::

     93 enum DsG4OpBoundaryProcessStatus {  Undefined,
     94                                   FresnelRefraction, FresnelReflection,
     95                                   TotalInternalReflection,
     96                                   LambertianReflection, LobeReflection,
     97                                   SpikeReflection, BackScattering,
     98                                   Absorption, Detection, NotAtBoundary,
     99                                   SameMaterial, StepTooSmall, NoRINDEX };
     100 
     101 class DsG4OpBoundaryProcess : public G4VDiscreteProcess
     102 {
     103 



:dybsvn:`source:dybgaudi/trunk/Simulation/DetSim/src/DsG4OpBoundaryProcess.cc`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    [blyth@cms01 src]$ diff ~/g4optical/src/G4OpBoundaryProcess.cc DsG4OpBoundaryProcess.cc
    ...
    < G4OpBoundaryProcess::PostStepDoIt(const G4Track& aTrack, const G4Step& aStep)
    ---
    > DsG4OpBoundaryProcess::PostStepDoIt(const G4Track& aTrack, const G4Step& aStep)
    179c181
    <           G4cerr << " G4OpBoundaryProcess/PostStepDoIt(): "
    ---
    >           G4cerr << " DsG4OpBoundaryProcess/PostStepDoIt(): "
    186a189,195
    >         if(theGlobalNormal.mag() == 0) {
    >           abNormalCounter++;
    >           std::cout << "Because of normal = 0, the number of the killed optical photons is " << abNormalCounter << std::endl;
    >           aParticleChange.ProposeTrackStatus(fStopAndKill);
    >           return G4VDiscreteProcess::PostStepDoIt(aTrack, aStep);
    >         }
    > 
    189c198
    <            G4cerr << " G4OpBoundaryProcess/PostStepDoIt(): "
    ---
    >            G4cerr << " DsG4OpBoundaryProcess/PostStepDoIt(): "
    301a311,336
    >                  if(OpticalSurface->GetName().contains("ESRAir")) {
    >                    G4double inciAngle = GetIncidentAngle();
    >                    //ESR in air
    >                    if(inciAngle*180./pi > 40) {
    >                      theReflectivity = (theReflectivity - 0.993) + 0.973572 + 9.53233e-04*(inciAngle*180./pi) - 1.22184e-05*((inciAngle*180./pi))*((inciAngle*180./pi));
    >                    }



:dybsvn:`source:dybgaudi/trunk/Simulation/DetSim/src/DsG4OpRayleigh.h`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    [blyth@cms01 src]$ diff ~/g4optical/include/G4OpRayleigh.hh DsG4OpRayleigh.h
    1a2,12
    > /**
    >  * \class DsG4OpRayleigh
    >  *
    >  * \brief A slightly modified version of G4OpRayleigh
    >  *
    >  * It is modified to make the Rayleigh Scattering happen with different waters defined in /dd/Material/
    >  *
    >  * This was taken from G4.9.1p1
    >  *
    >  * zhangh@bnl.gov on 8th, July
    >  */


:dybsvn:`source:dybgaudi/trunk/Simulation/DetSim/src/DsG4OpRayleigh.cc`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Specialization to handle `/dd/Materials/Water` etc.. names for  "Water"

::

    [blyth@cms01 src]$ diff ~/g4optical/src/G4OpRayleigh.cc DsG4OpRayleigh.cc
    ...
    < void G4OpRayleigh::BuildThePhysicsTable()
    ---
    > void DsG4OpRayleigh::BuildThePhysicsTable()
    226c238,241
    <                 if ((*theMaterialTable)[i]->GetName() == "Water")
    ---
    >                 if ((*theMaterialTable)[i]->GetName() == "/dd/Materials/Water" || 
    >                   (*theMaterialTable)[i]->GetName() == "/dd/Materials/OwsWater" ||
    >                   (*theMaterialTable)[i]->GetName() == "/dd/Materials/IwsWater"
    >                   )
    246c261
    < G4double G4OpRayleigh::GetMeanFreePath(const G4Track& aTrack,
    ---
    > G4double DsG4OpRayleigh::GetMeanFreePath(const G4Track& aTrack,
    257c272,275
    <         if (aMaterial->GetName() == "Water" && DefaultWater){
    ---
    >       if ((strcmp(aMaterial->GetName(), "/dd/Materials/Water") == 0 ||
    >            strcmp(aMaterial->GetName(), "/dd/Materials/OwsWater") == 0 ||
    >            strcmp(aMaterial->GetName(), "/dd/Materials/IwsWater") == 0 )
    >           && DefaultWater){
    294c312




