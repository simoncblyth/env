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

