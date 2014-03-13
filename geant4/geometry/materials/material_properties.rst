Material Properties
====================

Objective
----------

Persist material optical properties (some of which are wavelength dependent) 
into G4DAE COLLADA XML using extra tags, thus allowing access from pycollada 
and thence into Chroma.


To Checkout
------------

* http://wiki.opengatecollaboration.org/Users_Guide_V6.2:Generating_and_tracking_optical_photons


G4 Docs
--------

* http://geant4.web.cern.ch/geant4/UserDocumentation/UsersGuides/ForApplicationDeveloper/html/ch05s02.html#sect.PhysProc.Photo

::

    Example 5.4.  Optical properties added to a G4MaterialPropertiesTable and linked to a G4Material

    const G4int NUMENTRIES = 32;

    G4double ppckov[NUMENTRIES] = {2.034*eV, ......, 4.136*eV};
    G4double rindex[NUMENTRIES] = {1.3435, ......, 1.3608};
    G4double absorption[NUMENTRIES] = {344.8*cm, ......, 1450.0*cm];

    G4MaterialPropertiesTable *MPT = new G4MaterialPropertiesTable();

    MPT -> AddConstProperty("SCINTILLATIONYIELD",100./MeV);

    MPT -> AddProperty("RINDEX",ppckov,rindex,NUMENTRIES}->SetSpline(true);
    MPT -> AddProperty("ABSLENGTH",ppckov,absorption,NUMENTRIES}->SetSpline(true);

    scintillator -> SetMaterialPropertiesTable(MPT);


Interfacing to Chroma
------------------------


Many NuWa properties are defined for wavelengths 200-800nm 
but the chroma standard wavelengths start from 60nm, 
this leads to flat properties from 60-200nm 
using the 200nm value.  

That might cause problems ? 
Perhaps should adjust standard wavelengths to 200:800 nm ?
to match the range where NuWa properties are defined.

::

        g4pb:materials blyth$ ./material_properties.py g4_00.dae.6
        ...
        __dd__Materials__OwsWater0xabb2118.ABSLENGTH : (314) 
        [[ 196.00020975  273.208     ]
         [ 197.00020287  369.628     ]
         [ 198.00001706  491.566     ]
         ..., 
         [ 788.00206356  482.415     ]
         [ 790.00041213  486.647     ]
         [ 800.00157879  486.681     ]]
        [[  60.          273.20800781]
         [  80.          273.20800781]
         [ 100.          273.20800781]
         ..., 
         [ 760.          371.97369385]
         [ 780.          425.70596313]
         [ 800.          486.68099976]]


Check Length units for ABSLENGTH etc.. in G4 and Chroma
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* http://www.researchgate.net/publication/223664774_Water_absorption_length_measurement_with_the_ANTARES_optical_beacon_system

  * 50m in sea water 


::

    g4pb:materials blyth$ ./material_properties.py g4_00.dae.6 __dd__Materials__Water ABSLENGTH
        __dd__Materials__Water0xae9c3c0.ABSLENGTH : (314) 
    [[   196.00020975    273.208     ]
     [   197.00020287    369.628     ]
     [   198.00001706    491.566     ]
     [   199.0001295     602.2       ]
     [   200.00007207    691.563     ]
     [   201.00001245    772.754     ]
     [   202.00012819    831.749     ]
     [   202.99994217    907.747     ]
     [   203.99996821    967.594     ]
     [   205.00006561   1006.68      ]
     [   206.00009085   1033.76      ]
     [   207.00024322   1072.49      ]
     [   208.00003469   1102.7       ]
     [   209.000018     1143.13      ]
     [   210.00005874   1154.75      ]
     [   211.00001988   1198.39      ]
     [   212.00012428   1224.04      ]
     [   213.00023997   1254.03      ]
     [   214.00023246   1285.14      ]
     [   214.99996476   1321.42      ]
     [   216.00004997   1347.67      ]
     [   216.99998719   1383.31      ]
     [   218.00002581   1407.86      ]
     [   219.00004006   1432.05      ]
     [   220.00029221   1507.14      ]
     [   221.00026859   1537.25      ]
     ...
     [   395.0001809   18471.45      ]
     [   397.50017779  19007.41      ]
     [   400.00016996  19849.15      ]
     [   402.5001836   20634.24      ]
     [   405.00015542  21308.7       ]
     [   407.50018103  21907.26      ]
     [   410.00017189  22540.12      ]
     [   412.50015905  23146.44      ]
     [   415.00016127  23696.99      ]
     [   417.50016973  24222.97      ]
     [   420.00016016  24671.11      ]
     [   422.5001635   25069.66      ]
     [   425.00016687  25569.5       ]
     [   427.50019897  26073.52      ]
     [   430.00019796  26518.15      ]
     [   432.50017173  26991.2       ]
     [   435.00017115  27337.65      ]
     [   437.50019957  27494.64      ]
     [   440.00019405  27600.03      ]
     [   442.50021077  27645.51      ]
     [   445.00017598  27722.87      ]
     [   447.50018518  27602.66      ]
     [   450.00020209  27372.58      ]
     [   452.50019947  27469.29      ]
     [   455.00020947  27970.48      ]
     [   457.50019162  28461.27      ]
     [   460.00021623  28732.16      ]
     [   462.50017693  28962.23      ]
     [   465.00018182  29355.16      ]
     [   467.50017715  29720.12      ]
     [   470.00018757  29804.19      ]
     [   472.50019419  29966.68      ]         ## almost 30m
     [   475.0002047   29939.17      ]
     [   477.50019942  29723.06      ]
     [   480.00020395  29587.05      ]
     [   482.50019742  29616.74      ]
     [   485.00022368  29549.18      ]
     [   487.50018354  29214.26      ]
     [   490.00021452  29045.91      ]
     [   492.50019417  28379.45      ]
     [   495.00020074  27812.48      ]
     [   497.50020396  26750.18      ]
     [   500.00020035  26096.03      ]
     [   502.50021442  24773.3       ]
     [   505.00021763  23353.15      ]
     [   507.50022935  22279.68      ]
     [   510.00019325  20385.15      ]
     [   512.50020639  18714.45      ]
     [   515.00022713  18010.98      ]
     [   517.50024086  18012.75      ]
     [   520.00021759  17787.87      ]
     [   522.50024318  17659.41      ]
     [   525.00021354  17717.04      ]
     [   527.50022924  17462.57      ]
     [   530.00019741  17362.82      ]
     [   532.50021112  17056.22      ]
     [   535.00023487  16986.27      ]
     ...
     [   774.00175221    404.675     ]
     [   776.00248277    415.612     ]
     [   777.99893752    420.511     ]
     [   780.00078438    425.708     ]
     [   781.99816258    458.717     ]
     [   784.00083899    476.198     ]
     [   785.99885052    475.074     ]
     [   788.00206356    482.415     ]
     [   790.00041213    486.647     ]
     [   800.00157879    486.681     ]]





Chroma Units
~~~~~~~~~~~~

:: 

    g4pb:cuda blyth$ grep SPEED_OF_LIGHT *.*
    photon.h:            p.time += absorption_distance/(SPEED_OF_LIGHT/s.refractive_index1);
    photon.h:           p.time += scattering_distance/(SPEED_OF_LIGHT/s.refractive_index1);
    photon.h:    p.time += s.distance_to_boundary/(SPEED_OF_LIGHT/s.refractive_index1);
    physical_constants.h:#define SPEED_OF_LIGHT 299.792458f
    g4pb:cuda blyth$ 
    g4pb:cuda blyth$ 
    g4pb:cuda blyth$ vi physical_constants.h

    4 // mm/ns
    5 #define SPEED_OF_LIGHT 299.792458f

    g4pb:cuda blyth$ pwd
    /usr/local/env/chroma/chroma/cuda





G4 Examples
--------------

::

    [blyth@cms01 examples]$ find . -name '*.cc'  -exec grep -H MaterialProperty {} \;
    ./advanced/air_shower/src/UltraDetectorConstruction.cc:  MPT_Acrylic->AddProperty("ABSLENGTH", new G4MaterialPropertyVector()) ;
    ./advanced/human_phantom/src/G4HumanPhantomMaterial.cc:#include "G4MaterialPropertyVector.hh"
    ./advanced/brachytherapy/src/BrachyDetectorConstruction.cc:#include "G4MaterialPropertyVector.hh"
    ./advanced/brachytherapy/src/BrachyDetectorConstructionI.cc:#include "G4MaterialPropertyVector.hh"
    ./advanced/brachytherapy/src/BrachyMaterial.cc:#include "G4MaterialPropertyVector.hh"
    ./advanced/medical_linac/src/MedLinacHead.cc:#include "G4MaterialPropertyVector.hh"
    ./advanced/medical_linac/src/MedLinacTargetAndFilterDecorator.cc:#include "G4MaterialPropertyVector.hh"
    ./advanced/medical_linac/src/MedLinacDetectorConstruction.cc:#include "G4MaterialPropertyVector.hh"
    ./advanced/medical_linac/src/MedLinacMLCDecorator.cc:#include "G4MaterialPropertyVector.hh"
    ./advanced/Rich/src/RichTbMaterial.cc:#include "G4MaterialPropertyVector.hh"
    ./advanced/radioprotection/src/RemSimMaterial.cc:#include "G4MaterialPropertyVector.hh"
    ./extended/parallel/ExDiane/src/BrachyDetectorConstruction.cc:#include "G4MaterialPropertyVector.hh"
    ./extended/parallel/ExDiane/src/BrachyDetectorConstructionI.cc:#include "G4MaterialPropertyVector.hh"
    ./extended/parallel/ExDiane/src/BrachyDetectorConstructionIr.cc:#include "G4MaterialPropertyVector.hh"
    ./extended/parallel/ExDiane/src/BrachyDetectorConstructionLeipzig.cc:#include "G4MaterialPropertyVector.hh"
    ./extended/parallel/ExDiane/src/BrachyMaterial.cc:#include "G4MaterialPropertyVector.hh"
    [blyth@cms01 examples]$ pwd
    /data/env/local/dyb/trunk/external/build/LCG/geant4.9.2.p01/examples



Units
-------

As I am using internal Geant4 numbers outside of CLHEP/Geant4 so I need to peep behind
the curtain of the units and understand the raw numbers being persisted::

   eV = electronvolt 
   electronvolt = 1.e-6*megaelectronvolt 
   megaelectronvolt = 1. 

   eV = 1.e-6

So the units of the photon energies in G4MaterialPropertyVector are **MeV** 
and the length values are in **mm**.


Chroma optical props are expresses in wavelengths in  **nm**, so 

::

    [blyth@cms01 include]$ grep h_Plan *.hh
    G4PhysicalConstants.hh:using CLHEP::h_Planck;
    [blyth@cms01 include]$ vi G4PhysicalConstants.hh

     27 #include <CLHEP/Units/PhysicalConstants.h>
     ..
     34 using CLHEP::c_light;


`/data/env/local/dyb/trunk/external/clhep/2.0.4.2/i686-slc4-gcc34-dbg/include/CLHEP/Units/PhysicalConstants.h`::

     60 //
     61 // c   = 299.792458 mm/ns
     62 // c^2 = 898.7404 (mm/ns)^2 
     63 //
     64 static const double c_light   = 2.99792458e+8 * m/s;
     65 static const double c_squared = c_light * c_light;
     66 
     67 //
     68 // h     = 4.13566e-12 MeV*ns
     69 // hbar  = 6.58212e-13 MeV*ns
     70 // hbarc = 197.32705e-12 MeV*mm
     71 //
     72 static const double h_Planck      = 6.62606896e-34 * joule*s;


`/data/env/local/dyb/trunk/external/clhep/2.0.4.2/i686-slc4-gcc34-dbg/include/CLHEP/Units/SystemOfUnits.h`::


    054   static const double millimeter  = 1.;
    062   static const double meter  = 1000.*millimeter;
    095   static const double m  = meter;
    123   static const double nanosecond  = 1.;
    124   static const double second      = 1.e+9 *nanosecond;
    ...
    139   // Electric charge [Q]
    140   //
    141   static const double eplus = 1. ;// positron charge
    142   static const double e_SI  = 1.602176487e-19;// positron charge in coulomb
    143   static const double coulomb = eplus/e_SI;// coulomb = 6.24150 e+18 * eplus
    144 
    145   //
    146   // Energy [E]
    147   //
    148   static const double megaelectronvolt = 1. ;
    149   static const double     electronvolt = 1.e-6*megaelectronvolt;
    150   static const double kiloelectronvolt = 1.e-3*megaelectronvolt;
    151   static const double gigaelectronvolt = 1.e+3*megaelectronvolt;
    152   static const double teraelectronvolt = 1.e+6*megaelectronvolt;
    153   static const double petaelectronvolt = 1.e+9*megaelectronvolt;
    154 
    155   static const double joule = electronvolt/e_SI;// joule = 6.24150 e+12 * MeV
    156 
    157   // symbols
    158   static const double MeV = megaelectronvolt;
    159   static const double  eV = electronvolt;
    160   static const double keV = kiloelectronvolt;
    161   static const double GeV = gigaelectronvolt;
    162   static const double TeV = teraelectronvolt;
    163   static const double PeV = petaelectronvolt;

  * g4_mpv_x * 1.e6   



* http://dayabay.bnl.gov/dox/DetSim/html/classDsG4OpRayleigh.html

  * suggests convention for photon energies in G4MaterialPropertyVector is MeV
  * :google:`"G4MaterialPropertyVector" photon energy unit`





G4 Optical processes
---------------------

#. bulk absorption
#. Rayleigh scattering (inverse 4th power of wavelength: sky blue, sun yellow : as viewed thru atmosphere)
#. reflection/refraction at material boundaries


Definitions from Peter Gumplinger
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* http://hypernews.slac.stanford.edu/HyperNews/geant4/get/opticalphotons/488.html

#. The attenuation length is the length before any interaction happens to a photon, be it scattering or absorption.
#. In Geant4, you can have absoption that simply removes the photon (G4OpAbsorption), 
   or you can have 'wavelength shifting' where the original photon is removed with the subsequent emission of a WLS photon. 
   The absorption length for the two processes may be different. The WLS process is called G4OpWLS.


G4 Material Properties in NuWa
--------------------------------

::

    [blyth@cms01 dybgaudi]$ find . -name '*.cc' -exec grep -l G4Material {} \;
    ./Simulation/Historian/src/QueriableStepAction.cc
    ./Simulation/Historian/src/UnObserverStepAction.cc
    ./Simulation/Historian/src/HistorianStepAction.cc
    ./Simulation/DetSim/src/DsG4Scintillation.cc
    ./Simulation/DetSim/src/DsG4NeutronHPThermalScattering.cc
    ./Simulation/DetSim/src/DsG4OpBoundaryProcess.cc
    ./Simulation/DetSim/src/DsG4MuNuclearInteraction.cc
    ./Simulation/DetSim/src/DsPmtSensDet.cc
    ./Simulation/DetSim/src/DsG4Cerenkov.cc
    ./Simulation/DetSim/src/DsG4OpRayleigh.cc
    ./Simulation/DetSim/src/DsG4NeutronHPCapture.cc
    [blyth@cms01 dybgaudi]$ pwd
    /data/env/local/dyb/trunk/NuWa-trunk/dybgaudi


property key usage
~~~~~~~~~~~~~~~~~~~~

::

    [blyth@cms01 dybgaudi]$ find . -name '*.cc' -exec grep -H GetProperty {} \;
    ./Simulation/Historian/src/QueriableStepAction.cc:          aMaterialPropertiesTable->GetProperty("FASTCOMPONENT"); 
    ./Simulation/Historian/src/QueriableStepAction.cc:          aMaterialPropertiesTable->GetProperty("SLOWCOMPONENT");

    ./Simulation/DetSim/src/DsG4Scintillation.cc:        aMaterialPropertiesTable->GetProperty("FASTCOMPONENT"); 
    ./Simulation/DetSim/src/DsG4Scintillation.cc:        aMaterialPropertiesTable->GetProperty("SLOWCOMPONENT");
    ./Simulation/DetSim/src/DsG4Scintillation.cc:        aMaterialPropertiesTable->GetProperty("REEMISSIONPROB");
    ./Simulation/DetSim/src/DsG4Scintillation.cc:            Reemission_Prob->GetProperty(aTrack.GetKineticEnergy());
    ./Simulation/DetSim/src/DsG4Scintillation.cc:                aMaterialPropertiesTable->GetProperty("SCINTILLATIONYIELD");
    ./Simulation/DetSim/src/DsG4Scintillation.cc:            ScintillationYield = ptable->GetProperty(0);
    ./Simulation/DetSim/src/DsG4Scintillation.cc:                aMaterialPropertiesTable->GetProperty("RESOLUTIONSCALE");
    ./Simulation/DetSim/src/DsG4Scintillation.cc:                ResolutionScale = ptable->GetProperty(0);
    ./Simulation/DetSim/src/DsG4Scintillation.cc:        aMaterialPropertiesTable->GetProperty(FastTimeConstant.c_str());
    ./Simulation/DetSim/src/DsG4Scintillation.cc:        if (!ptable) ptable = aMaterialPropertiesTable->GetProperty("FASTTIMECONSTANT");
    ./Simulation/DetSim/src/DsG4Scintillation.cc:            fastTimeConstant = ptable->GetProperty(0);
    ./Simulation/DetSim/src/DsG4Scintillation.cc:        aMaterialPropertiesTable->GetProperty(SlowTimeConstant.c_str());
    ./Simulation/DetSim/src/DsG4Scintillation.cc:        if(!ptable) ptable = aMaterialPropertiesTable->GetProperty("SLOWTIMECONSTANT");
    ./Simulation/DetSim/src/DsG4Scintillation.cc:          slowTimeConstant = ptable->GetProperty(0);
    ./Simulation/DetSim/src/DsG4Scintillation.cc:            aMaterialPropertiesTable->GetProperty(strYieldRatio.c_str());
    ./Simulation/DetSim/src/DsG4Scintillation.cc:        if(!ptable) ptable = aMaterialPropertiesTable->GetProperty("YIELDRATIO");
    ./Simulation/DetSim/src/DsG4Scintillation.cc:            YieldRatio = ptable->GetProperty(0);
    ./Simulation/DetSim/src/DsG4Scintillation.cc:                aMaterialPropertiesTable->GetProperty("FASTCOMPONENT");
    ./Simulation/DetSim/src/DsG4Scintillation.cc:                    GetProperty();
    ./Simulation/DetSim/src/DsG4Scintillation.cc:                            GetProperty();
    ./Simulation/DetSim/src/DsG4Scintillation.cc:                aMaterialPropertiesTable->GetProperty("SLOWCOMPONENT");
    ./Simulation/DetSim/src/DsG4Scintillation.cc:                    GetProperty();
    ./Simulation/DetSim/src/DsG4Scintillation.cc:                            GetProperty();
    ./Simulation/DetSim/src/DsG4Scintillation.cc:                aMaterialPropertiesTable->GetProperty("REEMISSIONPROB");
    ./Simulation/DetSim/src/DsG4Scintillation.cc:                    GetProperty();
    ./Simulation/DetSim/src/DsG4Scintillation.cc:                            GetProperty();

    ./Simulation/DetSim/src/DsG4OpBoundaryProcess.cc:               Rindex = aMaterialPropertiesTable->GetProperty("RINDEX");
    ./Simulation/DetSim/src/DsG4OpBoundaryProcess.cc:               Rindex1 = Rindex->GetProperty(thePhotonMomentum);
    ./Simulation/DetSim/src/DsG4OpBoundaryProcess.cc:                  Rindex = aMaterialPropertiesTable->GetProperty("RINDEX");
    ./Simulation/DetSim/src/DsG4OpBoundaryProcess.cc:                     Rindex2 = Rindex->GetProperty(thePhotonMomentum);
    ./Simulation/DetSim/src/DsG4OpBoundaryProcess.cc:                      aMaterialPropertiesTable->GetProperty("REFLECTIVITY");
    ./Simulation/DetSim/src/DsG4OpBoundaryProcess.cc:                      aMaterialPropertiesTable->GetProperty("REALRINDEX");
    ./Simulation/DetSim/src/DsG4OpBoundaryProcess.cc:                      aMaterialPropertiesTable->GetProperty("IMAGINARYRINDEX");
    ./Simulation/DetSim/src/DsG4OpBoundaryProcess.cc:                          PropertyPointer->GetProperty(thePhotonMomentum);
    ./Simulation/DetSim/src/DsG4OpBoundaryProcess.cc:                          PropertyPointer1->GetProperty(thePhotonMomentum);
    ./Simulation/DetSim/src/DsG4OpBoundaryProcess.cc:                          PropertyPointer2->GetProperty(thePhotonMomentum);
    ./Simulation/DetSim/src/DsG4OpBoundaryProcess.cc:              aMaterialPropertiesTable->GetProperty("EFFICIENCY");
    ./Simulation/DetSim/src/DsG4OpBoundaryProcess.cc:                      PropertyPointer->GetProperty(thePhotonMomentum);
    ./Simulation/DetSim/src/DsG4OpBoundaryProcess.cc:               aMaterialPropertiesTable->GetProperty("SPECULARLOBECONSTANT");
    ./Simulation/DetSim/src/DsG4OpBoundaryProcess.cc:                        PropertyPointer->GetProperty(thePhotonMomentum);
    ./Simulation/DetSim/src/DsG4OpBoundaryProcess.cc:               aMaterialPropertiesTable->GetProperty("SPECULARSPIKECONSTANT");
    ./Simulation/DetSim/src/DsG4OpBoundaryProcess.cc:                        PropertyPointer->GetProperty(thePhotonMomentum);
    ./Simulation/DetSim/src/DsG4OpBoundaryProcess.cc:               aMaterialPropertiesTable->GetProperty("BACKSCATTERCONSTANT");
    ./Simulation/DetSim/src/DsG4OpBoundaryProcess.cc:                        PropertyPointer->GetProperty(thePhotonMomentum);
    ./Simulation/DetSim/src/DsG4OpBoundaryProcess.cc:                 Rindex = aMaterialPropertiesTable->GetProperty("RINDEX");
    ./Simulation/DetSim/src/DsG4OpBoundaryProcess.cc:                 Rindex2 = Rindex->GetProperty(thePhotonMomentum);

    ./Simulation/DetSim/src/DsPmtSensDet.cc:        G4MaterialPropertyVector* qevec = mattab->GetProperty(m_qeffParamName.c_str());
    ./Simulation/DetSim/src/DsPmtSensDet.cc:          return qevec->GetProperty(energy);

    ./Simulation/DetSim/src/DsG4Cerenkov.cc:                aMaterialPropertiesTable->GetProperty("RINDEX"); 
    ./Simulation/DetSim/src/DsG4Cerenkov.cc:            sampledRI = Rindex->GetProperty(sampledEnergy);
    ./Simulation/DetSim/src/DsG4Cerenkov.cc:                           aMaterialPropertiesTable->GetProperty("RINDEX");
    ./Simulation/DetSim/src/DsG4Cerenkov.cc:                                           GetProperty();
    ./Simulation/DetSim/src/DsG4Cerenkov.cc:                                                GetProperty();
    ./Simulation/DetSim/src/DsG4Cerenkov.cc:                     Rindex = aMaterialPropertiesTable->GetProperty("RINDEX");
    ./Simulation/DetSim/src/DsG4Cerenkov.cc:        // GetProperty() methods of the G4MaterialPropertiesTable and
    ./Simulation/DetSim/src/DsG4Cerenkov.cc:  G4MaterialPropertyVector* qevec = bialkali->GetMaterialPropertiesTable()->GetProperty("EFFICIENCY");
    ./Simulation/DetSim/src/DsG4Cerenkov.cc:  return qevec->GetProperty(energy);

    ./Simulation/DetSim/src/DsG4OpRayleigh.cc:                            aMaterialPropertiesTable->GetProperty("RAYLEIGH");
    ./Simulation/DetSim/src/DsG4OpRayleigh.cc:                   aMaterialPropertyTable->GetProperty("RAYLEIGH");
    ./Simulation/DetSim/src/DsG4OpRayleigh.cc:                                    GetProperty(thePhotonEnergy);
    ./Simulation/DetSim/src/DsG4OpRayleigh.cc:        G4MaterialPropertyVector* Rindex = aMPT->GetProperty("RINDEX");
    ./Simulation/DetSim/src/DsG4OpRayleigh.cc:                refraction_index = Rindex->GetProperty();
    [blyth@cms01 dybgaudi]$ 




NuWa surface properties 
~~~~~~~~~~~~~~~~~~~~~~~~

::

    [blyth@cms01 dybgaudi]$ find . -name '*.cc' -exec grep -H Surface {} \;
    ./Simulation/GenTools/src/components/GtPositionerTool.cc:    if ("Surface" == m_strategy) {
    ./Simulation/GenTools/src/components/GtPositionerTool.cc:        fatal() << "Surface strategy not yet supported" << endreq;
    ./Simulation/GenTools/src/components/GtRockGammaTool.cc:   m_totalSurfaceArea(0),
    ./Simulation/GenTools/src/components/GtRockGammaTool.cc:  m_totalSurfaceArea = 0;
    ./Simulation/GenTools/src/components/GtRockGammaTool.cc:    m_totalSurfaceArea += m_walls[wallIdx]->area();
    ./Simulation/GenTools/src/components/GtRockGammaTool.cc:  debug() << "Total surface area: " << m_totalSurfaceArea << endreq;
    ./Simulation/GenTools/src/components/GtRockGammaTool.cc:  double randArea = rand*m_totalSurfaceArea;
    ./Simulation/DetSim/src/DsG4OpBoundaryProcess.cc://              1998-11-07 - NULL OpticalSurface pointer before use
    ./Simulation/DetSim/src/DsG4OpBoundaryProcess.cc://                           G4OpticalSurface class ( by Fan Lei)
    ./Simulation/DetSim/src/DsG4OpBoundaryProcess.cc:                        ->GetSurfaceTolerance();
    ./Simulation/DetSim/src/DsG4OpBoundaryProcess.cc:        G4SurfaceType type = dielectric_dielectric;
    ./Simulation/DetSim/src/DsG4OpBoundaryProcess.cc:        OpticalSurface = NULL;
    ./Simulation/DetSim/src/DsG4OpBoundaryProcess.cc:        G4LogicalSurface* Surface = NULL;
    ./Simulation/DetSim/src/DsG4OpBoundaryProcess.cc:        Surface = G4LogicalBorderSurface::GetSurface
    ./Simulation/DetSim/src/DsG4OpBoundaryProcess.cc:        if (Surface == NULL){
    ./Simulation/DetSim/src/DsG4OpBoundaryProcess.cc:       Surface = G4LogicalSkinSurface::GetSurface
    ./Simulation/DetSim/src/DsG4OpBoundaryProcess.cc:       if(Surface == NULL)
    ./Simulation/DetSim/src/DsG4OpBoundaryProcess.cc:         Surface = G4LogicalSkinSurface::GetSurface
    ./Simulation/DetSim/src/DsG4OpBoundaryProcess.cc:       Surface = G4LogicalSkinSurface::GetSurface
    ./Simulation/DetSim/src/DsG4OpBoundaryProcess.cc:       if(Surface == NULL)
    ./Simulation/DetSim/src/DsG4OpBoundaryProcess.cc:         Surface = G4LogicalSkinSurface::GetSurface
    ./Simulation/DetSim/src/DsG4OpBoundaryProcess.cc:   if (Surface) OpticalSurface = 
    ./Simulation/DetSim/src/DsG4OpBoundaryProcess.cc:           dynamic_cast <G4OpticalSurface*> (Surface->GetSurfaceProperty());
    ./Simulation/DetSim/src/DsG4OpBoundaryProcess.cc:   if (OpticalSurface) {
    ./Simulation/DetSim/src/DsG4OpBoundaryProcess.cc:           type      = OpticalSurface->GetType();
    ./Simulation/DetSim/src/DsG4OpBoundaryProcess.cc:      theModel  = OpticalSurface->GetModel();
    ./Simulation/DetSim/src/DsG4OpBoundaryProcess.cc:      theFinish = OpticalSurface->GetFinish();
    ./Simulation/DetSim/src/DsG4OpBoundaryProcess.cc:      aMaterialPropertiesTable = OpticalSurface->
    ./Simulation/DetSim/src/DsG4OpBoundaryProcess.cc:                 if(OpticalSurface->GetName().contains("ESRAir")) {
    ./Simulation/DetSim/src/DsG4OpBoundaryProcess.cc:      if (OpticalSurface) sigma_alpha = OpticalSurface->GetSigmaAlpha();
    ./Simulation/DetSim/src/DsG4OpBoundaryProcess.cc:      if (OpticalSurface) polish = OpticalSurface->GetPolish();
    ./Validation/GeometryVal/src/GeometryVal.cc:    if ("Surface" == m_strategy) {
    ./Validation/GeometryVal/src/GeometryVal.cc:        fatal() << "Surface strategy not yet supported" << endreq;
    ./Reconstruction/Likelihood/src/LikelihoodTool.cc:#include "DetDesc/Surface.h"
    ./Reconstruction/AdRec/src/components/ExpQCalcTool.cc:#include "DetDesc/Surface.h"
    ./Reconstruction/AdRec/src/components/ExpQCalcTool.cc:    std::string topESR_location = "/dd/Geometry/AdDetails/AdSurfacesAll/ESRAirSurfaceTop";
    ./Reconstruction/AdRec/src/components/ExpQCalcTool.cc:    std::string botESR_location = "/dd/Geometry/AdDetails/AdSurfacesAll/ESRAirSurfaceBot";
    ./Reconstruction/AdRec/src/components/ExpQCalcTool.cc:    Surface* esrtop = GaudiCommon<AlgTool>::get<Surface>(dds, topESR_location);
    ./Reconstruction/AdRec/src/components/ExpQCalcTool.cc:    Surface* esrbot = GaudiCommon<AlgTool>::get<Surface>(dds, botESR_location);
    ./Reconstruction/AdRec/src/components/ExpQCalcTool.cc:    Surface::Tables& esrtop_tab = esrtop->tabulatedProperties();
    ./Reconstruction/AdRec/src/components/ExpQCalcTool.cc:    Surface::Tables& esrbot_tab = esrbot->tabulatedProperties();
    ./Reconstruction/AdRec/src/components/ExpQCalcTool.cc:    Surface::Tables::const_iterator sfIter; 
    ./Reconstruction/AdRec/src/components/QMLFTool.cc:#include "DetDesc/Surface.h"
    ./Reconstruction/AdRec/src/components/QMLFTool.cc:      m_opPara.m_topRef = meanOpticalPara<Surface>(
    ./Reconstruction/AdRec/src/components/QMLFTool.cc:          "/dd/Geometry/AdDetails/AdSurfacesAll/ESRAirSurfaceTop",
    ./Reconstruction/AdRec/src/components/QMLFTool.cc:      m_opPara.m_botRef = meanOpticalPara<Surface>(
    ./Reconstruction/AdRec/src/components/QMLFTool.cc:          "/dd/Geometry/AdDetails/AdSurfacesAll/ESRAirSurfaceBot",
    ./Reconstruction/PoolRec/MuonPoolEvtDsp/src/components/PointSurfacePosition.cc:#include "PointSurfacePosition.h"
    ./Reconstruction/PoolRec/MuonPoolEvtDsp/src/components/PointSurfacePosition.cc:void PointSurfacePosition::Set(double x, double y, double z, int wall )
    ./Reconstruction/PoolRec/MuonPoolEvtDsp/src/components/FarPoolEvtDsp.cc:#include "PointSurfacePosition.h"
    ./Reconstruction/PoolRec/MuonPoolEvtDsp/src/components/FarPoolEvtDsp.cc:   vector<PointSurfacePosition> prealv;
    ./Reconstruction/PoolRec/MuonPoolEvtDsp/src/components/FarPoolEvtDsp.cc:   vector<PointSurfacePosition>::const_iterator it_prealv;
    [blyth@cms01 dybgaudi]$ 




Optical Surface modelling in G4
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


`NuWa-trunk/dybgaudi/Simulation/DetSim/src/DsG4OpBoundaryProcess.cc`::

    227         theModel = glisur;
    228         theFinish = polished;
    229 
    230         G4SurfaceType type = dielectric_dielectric;
    231 
    232         Rindex = NULL;
    233         OpticalSurface = NULL;
    234 
    235         G4LogicalSurface* Surface = NULL;
    236 
    237         Surface = G4LogicalBorderSurface::GetSurface
    238               (pPreStepPoint ->GetPhysicalVolume(),
    239                pPostStepPoint->GetPhysicalVolume());
    240 
    241         if (Surface == NULL){
    242       G4bool enteredDaughter=(pPostStepPoint->GetPhysicalVolume()
    243                   ->GetMotherLogical() ==
    244                   pPreStepPoint->GetPhysicalVolume()
    245                   ->GetLogicalVolume());
    246       if(enteredDaughter){
    247         Surface = G4LogicalSkinSurface::GetSurface
    248           (pPostStepPoint->GetPhysicalVolume()->
    249            GetLogicalVolume());
    250         if(Surface == NULL)
    251           Surface = G4LogicalSkinSurface::GetSurface
    252           (pPreStepPoint->GetPhysicalVolume()->
    253            GetLogicalVolume());
    254       }
    255       else {
    256         Surface = G4LogicalSkinSurface::GetSurface
    257           (pPreStepPoint->GetPhysicalVolume()->
    258            GetLogicalVolume());
    259         if(Surface == NULL)
    260           Surface = G4LogicalSkinSurface::GetSurface
    261           (pPostStepPoint->GetPhysicalVolume()->
    262            GetLogicalVolume());
    263       }
    264     }
    265 
    266     if (Surface) OpticalSurface =
    267            dynamic_cast <G4OpticalSurface*> (Surface->GetSurfaceProperty());
    268 
    269     if (OpticalSurface) {
    270 
    271            type      = OpticalSurface->GetType();
    272        theModel  = OpticalSurface->GetModel();
    273        theFinish = OpticalSurface->GetFinish();
    274 
    275        aMaterialPropertiesTable = OpticalSurface->
    276                     GetMaterialPropertiesTable();


GDML persisting optical surface properties
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Old G4::

    [blyth@cms01 src]$ pwd
    /data/env/local/dyb/trunk/external/build/LCG/geant4.9.2.p01/source/persistency/gdml/src

    [blyth@cms01 src]$ grep G4Optical *.cc
    G4GDMLReadSolids.cc:   G4OpticalSurfaceModel model; 
    G4GDMLReadSolids.cc:   G4OpticalSurfaceFinish finish;
    G4GDMLReadSolids.cc:   new G4OpticalSurface(name,model,finish,type,value);
    [blyth@cms01 src]$ 
    [blyth@cms01 src]$ grep G4Optical ../include/*.hh
    ../include/G4GDMLReadSolids.hh:#include "G4OpticalSurface.hh"
    [blyth@cms01 src]$ 

New G4::

    g4pb:src blyth$ pwd
    /usr/local/env/geant4/geant4.10.00.b01/source/persistency/gdml/src

    g4pb:src blyth$ grep G4Optical *.cc
    G4GDMLReadSolids.cc:#include "G4OpticalSurface.hh"
    G4GDMLReadSolids.cc:   G4OpticalSurfaceModel model; 
    G4GDMLReadSolids.cc:   G4OpticalSurfaceFinish finish;
    G4GDMLReadSolids.cc:   new G4OpticalSurface(name,model,finish,type,value);
    G4GDMLWriteSolids.cc:#include "G4OpticalSurface.hh"
    G4GDMLWriteSolids.cc:                    const G4OpticalSurface* const surf)
    G4GDMLWriteSolids.cc:   G4OpticalSurfaceModel smodel = surf->GetModel();
    G4GDMLWriteStructure.cc:#include "G4OpticalSurface.hh"
    G4GDMLWriteStructure.cc:     const G4OpticalSurface* opsurf =
    G4GDMLWriteStructure.cc:       dynamic_cast<const G4OpticalSurface*>(psurf);
    G4GDMLWriteStructure.cc:     const G4OpticalSurface* opsurf =
    G4GDMLWriteStructure.cc:       dynamic_cast<const G4OpticalSurface*>(psurf);
    G4GDMLWriteStructure.cc:   const G4OpticalSurface* osurf = dynamic_cast<const G4OpticalSurface*>(psurf);
    G4GDMLWriteStructure.cc:   std::vector<const G4OpticalSurface*>::const_iterator pos;
    g4pb:src blyth$ 
    g4pb:src blyth$ grep G4Optical ../include/*.hh
    ../include/G4GDMLWriteSolids.hh:class G4OpticalSurface;
    ../include/G4GDMLWriteSolids.hh:                    const G4OpticalSurface* const);
    ../include/G4GDMLWriteStructure.hh:class G4OpticalSurface;
    ../include/G4GDMLWriteStructure.hh:   std::vector<const G4OpticalSurface*> opt_vec;
    g4pb:src blyth$ 


Reading new GDML persisting code, it looks like the surface property tables are not persisted, 
although the surface names are::

    g4pb:src blyth$ grep GetMaterialPropertiesTable *.cc
    G4GDMLReadMaterials.cc:   G4MaterialPropertiesTable* matprop=material->GetMaterialPropertiesTable();
    G4GDMLWriteMaterials.cc:   if (materialPtr->GetMaterialPropertiesTable())
    G4GDMLWriteMaterials.cc:   G4MaterialPropertiesTable* ptable = mat->GetMaterialPropertiesTable();
    g4pb:src blyth$ 



From the GDML volume traverse `G4GDMLWriteStructure.cc`::

    359 G4Transform3D G4GDMLWriteStructure::
    360 TraverseVolumeTree(const G4LogicalVolume* const volumePtr, const G4int depth)
    361 {
    ...

    480       else   // Is it a physvol?
    481       {
    482          G4RotationMatrix rot;
    483 
    484          if (physvol->GetFrameRotation() != 0)
    485          {
    486            rot = *(physvol->GetFrameRotation());
    487          }
    488          G4Transform3D P(rot,physvol->GetObjectTranslation());
    489          PhysvolWrite(volumeElement,physvol,invR*P*daughterR,ModuleName);
    490       }
    491       BorderSurfaceCache(GetBorderSurface(physvol));
    492    }
    493 
    494    structureElement->appendChild(volumeElement);
    495      // Append the volume AFTER traversing the children so that
    496      // the order of volumes will be correct!
    497 
    498    VolumeMap()[volumePtr] = R;
    499 
    500    AddExtension(volumeElement, volumePtr);
    501      // Add any possible user defined extension attached to a volume
    502 
    503    AddMaterial(volumePtr->GetMaterial());
    504      // Add the involved materials and solids!
    505 
    506    AddSolid(solidPtr);
    507 
    508    SkinSurfaceCache(GetSkinSurface(volumePtr));
    509 
    510    return R;
    511 }
    ...
    294 const G4LogicalSkinSurface*
    295 G4GDMLWriteStructure::GetSkinSurface(const G4LogicalVolume* const lvol)
    296 {
    297   G4LogicalSkinSurface* surf = 0;
    298   G4int nsurf = G4LogicalSkinSurface::GetNumberOfSkinSurfaces();
    299   if (nsurf)
    300   {
    301     const G4LogicalSkinSurfaceTable* stable =
    302           G4LogicalSkinSurface::GetSurfaceTable();
    303     std::vector<G4LogicalSkinSurface*>::const_iterator pos;
    304     for (pos = stable->begin(); pos != stable->end(); pos++)
    305     {
    306       if (lvol == (*pos)->GetLogicalVolume())
    307       {
    308         surf = *pos; break;
    309       }
    310     }
    311   }
    312   return surf;
    313 }
    314 
    315 const G4LogicalBorderSurface*
    316 G4GDMLWriteStructure::GetBorderSurface(const G4VPhysicalVolume* const pvol)
    317 {
    318   G4LogicalBorderSurface* surf = 0;
    319   G4int nsurf = G4LogicalBorderSurface::GetNumberOfBorderSurfaces();
    320   if (nsurf)
    321   {
    322     const G4LogicalBorderSurfaceTable* btable =
    323           G4LogicalBorderSurface::GetSurfaceTable();
    324     std::vector<G4LogicalBorderSurface*>::const_iterator pos;
    325     for (pos = btable->begin(); pos != btable->end(); pos++)
    326     {
    327       if (pvol == (*pos)->GetVolume1())  // just the first in the couple 
    328       {                                  // is enough
    329         surf = *pos; break;
    330       }
    331     }
    332   }
    333   return surf;




Same in g4ten::

    [blyth@belle7 src]$ grep GetMaterialPropertiesTable *.cc
    G4GDMLReadMaterials.cc:   G4MaterialPropertiesTable* matprop=material->GetMaterialPropertiesTable();
    G4GDMLWriteMaterials.cc:   if (materialPtr->GetMaterialPropertiesTable())
    G4GDMLWriteMaterials.cc:   G4MaterialPropertiesTable* ptable = mat->GetMaterialPropertiesTable();
    [blyth@belle7 src]$ 
    [blyth@belle7 src]$ pwd
    /data1/env/local/env/geant4/geant4.10.00/source/persistency/gdml/src


Many diffs but not relevant to G4DAE::

    [blyth@belle7 src]$ diff -r --brief /data1/env/local/env/geant4/geant4.10.00.b01/source/persistency/gdml /data1/env/local/env/geant4/geant4.10.00/source/persistency/gdml 
    Files /data1/env/local/env/geant4/geant4.10.00.b01/source/persistency/gdml/History and /data1/env/local/env/geant4/geant4.10.00/source/persistency/gdml/History differ
    Files /data1/env/local/env/geant4/geant4.10.00.b01/source/persistency/gdml/include/G4GDMLParameterisation.hh and /data1/env/local/env/geant4/geant4.10.00/source/persistency/gdml/include/G4GDMLParameterisation.hh differ
    Files /data1/env/local/env/geant4/geant4.10.00.b01/source/persistency/gdml/include/G4GDMLReadParamvol.hh and /data1/env/local/env/geant4/geant4.10.00/source/persistency/gdml/include/G4GDMLReadParamvol.hh differ
    Files /data1/env/local/env/geant4/geant4.10.00.b01/source/persistency/gdml/include/G4GDMLReadSolids.hh and /data1/env/local/env/geant4/geant4.10.00/source/persistency/gdml/include/G4GDMLReadSolids.hh differ
    Files /data1/env/local/env/geant4/geant4.10.00.b01/source/persistency/gdml/include/G4GDMLWriteParamvol.hh and /data1/env/local/env/geant4/geant4.10.00/source/persistency/gdml/include/G4GDMLWriteParamvol.hh differ
    Files /data1/env/local/env/geant4/geant4.10.00.b01/source/persistency/gdml/include/G4GDMLWriteSolids.hh and /data1/env/local/env/geant4/geant4.10.00/source/persistency/gdml/include/G4GDMLWriteSolids.hh differ
    Files /data1/env/local/env/geant4/geant4.10.00.b01/source/persistency/gdml/schema/gdml_parameterised.xsd and /data1/env/local/env/geant4/geant4.10.00/source/persistency/gdml/schema/gdml_parameterised.xsd differ
    Files /data1/env/local/env/geant4/geant4.10.00.b01/source/persistency/gdml/schema/gdml_solids.xsd and /data1/env/local/env/geant4/geant4.10.00/source/persistency/gdml/schema/gdml_solids.xsd differ
    Files /data1/env/local/env/geant4/geant4.10.00.b01/source/persistency/gdml/src/G4GDMLParameterisation.cc and /data1/env/local/env/geant4/geant4.10.00/source/persistency/gdml/src/G4GDMLParameterisation.cc differ
    Files /data1/env/local/env/geant4/geant4.10.00.b01/source/persistency/gdml/src/G4GDMLReadParamvol.cc and /data1/env/local/env/geant4/geant4.10.00/source/persistency/gdml/src/G4GDMLReadParamvol.cc differ
    Files /data1/env/local/env/geant4/geant4.10.00.b01/source/persistency/gdml/src/G4GDMLReadSolids.cc and /data1/env/local/env/geant4/geant4.10.00/source/persistency/gdml/src/G4GDMLReadSolids.cc differ
    Files /data1/env/local/env/geant4/geant4.10.00.b01/source/persistency/gdml/src/G4GDMLWriteParamvol.cc and /data1/env/local/env/geant4/geant4.10.00/source/persistency/gdml/src/G4GDMLWriteParamvol.cc differ
    Files /data1/env/local/env/geant4/geant4.10.00.b01/source/persistency/gdml/src/G4GDMLWriteSolids.cc and /data1/env/local/env/geant4/geant4.10.00/source/persistency/gdml/src/G4GDMLWriteSolids.cc differ
    [blyth@belle7 src]$ 



G4LogicalSkinSurface and G4LogicalBorderSurface
-------------------------------------------------

::

    [blyth@cms01 source]$ find . -name G4LogicalSkinSurface.hh
    ./geometry/volumes/include/G4LogicalSkinSurface.hh
    [blyth@cms01 source]$ vi geometry/volumes/include/G4LogicalSkinSurface.hh

    34 // A Logical Surface class for the surface surrounding a single logical
    35 // volume.


    [blyth@cms01 source]$ find . -name G4LogicalBorderSurface.hh
    ./geometry/volumes/include/G4LogicalBorderSurface.hh
    [blyth@cms01 source]$ vi geometry/volumes/include/G4LogicalBorderSurface.hh

    34 // A Logical Surface class for surfaces defined by the boundary
    35 // of two physical volumes.




Some Hardcoded reflectivity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Hardcoded ESRAir reflectivity as function of incident angle. **this needs to be duplicated Chroma side GPU** 

::

    269     if (OpticalSurface) {
    270 
    271            type      = OpticalSurface->GetType();
    272        theModel  = OpticalSurface->GetModel();
    273        theFinish = OpticalSurface->GetFinish();
    274 
    275        aMaterialPropertiesTable = OpticalSurface->
    276                     GetMaterialPropertiesTable();
    277 
    278            if (aMaterialPropertiesTable) {
    279 
    280               if (theFinish == polishedbackpainted ||
    281                   theFinish == groundbackpainted ) {
    282                   Rindex = aMaterialPropertiesTable->GetProperty("RINDEX");
    283               if (Rindex) {
    284                      Rindex2 = Rindex->GetProperty(thePhotonMomentum);
    285                   }
    286                   else {
    287              theStatus = NoRINDEX;
    288                      aParticleChange.ProposeTrackStatus(fStopAndKill);
    289                      return G4VDiscreteProcess::PostStepDoIt(aTrack, aStep);
    290                   }
    291               }
    292 
    293               G4MaterialPropertyVector* PropertyPointer;
    294               G4MaterialPropertyVector* PropertyPointer1;
    295               G4MaterialPropertyVector* PropertyPointer2;
    296 
    297               PropertyPointer =
    298                       aMaterialPropertiesTable->GetProperty("REFLECTIVITY");
    299               PropertyPointer1 =
    300                       aMaterialPropertiesTable->GetProperty("REALRINDEX");
    301               PropertyPointer2 =
    302                       aMaterialPropertiesTable->GetProperty("IMAGINARYRINDEX");
    303 
    304               iTE = 1;
    305               iTM = 1;
    306 
    307               if (PropertyPointer) {
    /// REFLECTIVITY provided
    308 
    309                  theReflectivity =
    310                           PropertyPointer->GetProperty(thePhotonMomentum);
    311                  if(OpticalSurface->GetName().contains("ESRAir")) {
    312                    G4double inciAngle = GetIncidentAngle();
    313                    //ESR in air
    314                    if(inciAngle*180./pi > 40) {
    315                      theReflectivity = (theReflectivity - 0.993) + 0.973572 + 9.53233e-04*(inciAngle*180./pi) - 1.22184e-05*((inciAngle*180./pi))*((inciAngle*180./pi));
    316                    }
    ...
    337 
    338               } else if (PropertyPointer1 && PropertyPointer2) {
    ///      REALRINDEX and IMAGINARYRINDEX provided  
    339 
    340                  G4double RealRindex =
    341                           PropertyPointer1->GetProperty(thePhotonMomentum);
    342                  G4double ImaginaryRindex =
    343                           PropertyPointer2->GetProperty(thePhotonMomentum);
    344 
    345                  // calculate FacetNormal
    346                  if ( theFinish == ground ) {
    347                     theFacetNormal =
    348                               GetFacetNormal(OldMomentum, theGlobalNormal);
    349                  } else {
    350                     theFacetNormal = theGlobalNormal;
    351                  }
    352 



/data/env/local/dyb/trunk/NuWa-trunk/dybgaudi/Detector/XmlDetDesc/DDDB/AdDetails/surfaces.xml::

     09   <catalog name="AdSurfacesAll">
     10     <surfaceref href="#RSOilSurface"/>  <!--Radial Shield-->
     11     <surfaceref href="#ESRAirSurfaceTop"/>
     12     <surfaceref href="#ESRAirSurfaceBot"/>
     13     <surfaceref href="#SSTOilSurface"/>
     14     <surfaceref href="#AdCableTraySurface"/>
     15   </catalog>
     16   <catalog name="AdSurfacesNear">
     17     <surfaceref href="#SSTWaterSurfaceNear1"/>
     18     <surfaceref href="#SSTWaterSurfaceNear2"/>
     19   </catalog>
     20   <catalog name="AdSurfacesFar">
     21     <surfaceref href="#SSTWaterSurfaceFar1"/>
     22     <surfaceref href="#SSTWaterSurfaceFar2"/>
     23     <surfaceref href="#SSTWaterSurfaceFar3"/>
     24     <surfaceref href="#SSTWaterSurfaceFar4"/>
     25   </catalog>
     26 
     27   <catalog name="AdTabProperties">
     28     <tabpropertyref href="properties.xml#RSOilReflectivity"/> <!--Radial Shield-->
     29     <tabpropertyref href="properties.xml#RSOilSpecularLobe"/> <!--Radial Shield-->
     30     <tabpropertyref href="properties.xml#RSOilSpecularSpike"/> <!--Radial Shield-->
     31     <tabpropertyref href="properties.xml#RSOilBackScattering"/> <!--Radial Shield-->
     32     <tabpropertyref href="properties.xml#ESRAirReflectivity"/>
     33     <tabpropertyref href="properties.xml#ESRAirSpecularLobe"/>
     34     <tabpropertyref href="properties.xml#ESRAirSpecularSpike"/>
     35     <tabpropertyref href="properties.xml#ESRAirBackScattering"/>
     36     <tabpropertyref href="properties.xml#SSTOilReflectivity"/>
     37     <tabpropertyref href="properties.xml#SSTWaterReflectivity"/>
     38   </catalog>
     39 
     40 
     41 
     42   <!-- Surfaces -->
     43 
     44   <!-- Reflector top and bottom -->
     45 
     46   <surface name="ESRAirSurfaceTop"
     47        model="unified"
     48        finish="polished"
     49        type="dielectric_metal"
     50        value="0.0"
     51        volfirst="/dd/Geometry/AdDetails/lvTopReflector#pvTopRefGap"
     52        volsecond="/dd/Geometry/AdDetails/lvTopRefGap#pvTopESR">
     53     <tabprops address="/dd/Geometry/AdDetails/AdTabProperties/ESRAirReflectivity"/>
     54   </surface>





/data/env/local/dyb/trunk/NuWa-trunk/dybgaudi/Detector/XmlDetDesc/DDDB/Parameters/surfaces.xml::

     09 <!-- Geant4's G4OpticalSurface enums -->
     10 <parameter name="polished" value="0"/>
     11 <parameter name="polishedfrontpainted" value="1" />
     12 <parameter name="polishedbackpainted" value="2" />
     13 <parameter name="ground" value="3" />
     14 <parameter name="groundfrontpainted" value="4" />
     15 <parameter name="groundbackpainted" value="5" />
     16 
     17 <parameter name="dielectric_metal" value="0" />
     18 <parameter name="dielectric_dielectric" value="1" />
     19 
     20 <parameter name="glisur" value="0" />
     21 <parameter name="unified" value="1" />



/data/env/local/dyb/trunk/NuWa-trunk/dybgaudi/Detector/XmlDetDesc/DDDB/AdDetails/properties.xml::

     08   <tabproperty name="ESRAirReflectivity"
     09                type="REFLECTIVITY"
     10                xunit="eV"
     11                yunit=""
     12                xaxis="Energy"
     13                yaxis="Reflectivity">
     14     1.55      0.98505
     15     1.63      0.98406
     16     1.68      0.96723
     17     1.72      0.9702
     18     1.77      0.97119
     19     1.82      0.96624
     20     1.88      0.95139
     21     1.94      0.98307
     22     2.00      0.9801
     23     2.07      0.98901
     24     2.14      0.98505
     25     2.21      0.96525
     26     2.30      0.97614
     27     2.38      0.97812
     28     2.48      0.97515
     29     2.58      0.96525
     30     2.70      0.96624
     31     2.82      0.96129
     32     2.95      0.95832
     33     3.10      0.95733
     34     3.26      0.73656
     35     3.44      0.11583
     36     3.65      0.10395
     37     3.88      0.11682
     38     4.13      0.14256
     39     4.43      0.1188
     40     4.77      0.18018
     41     4.96      0.21384
     42     6.20      0.0099
     43     10.33     0.0099
     44     15.5      0.0099
     45   </tabproperty> <!-- reflectivity -->
     46 
     47   <tabproperty name="ESRAirSpecularLobe"
     48                type="SPECULARLOBECONSTANT"
     49                xunit="eV"
     50                yunit=""
     51                xaxis="Energy"
     52                yaxis="Specularlobe">
     53             1.55      1.
     54             1.63      1.
     55             1.68      1.
     ..


/data/env/local/dyb/trunk/NuWa-trunk/dybgaudi/Detector/XmlDetDesc/DDDB/materials/water.xml::

     10 
     11 
     12     <!-- This is taken from G4dyb's MaterialProperties.xml and massaged to fit -->
     13     <tabproperty name="WaterAbsorptionLength"
     14          type="ABSLENGTH"
     15          xunit="eV"
     16          yunit="cm"
     17          xaxis="PhotonEnergy"
     18          yaxis="AbsorptionLength">
     19 
     20          1.5498024  48.66809
     21          1.5694201  48.66475
     22          1.5734035  48.24148
     23          1.5774070  47.50742
     24          1.5814310  47.61985
     25          1.5854756  45.87167
     ..




Key discoverability
---------------------

* http://www-zeuthen.desy.de/geant4/geant4.9.3.b01/G4MaterialPropertiesTable_8hh-source.html

4.9.3 has public map accessors::

    00119   public:  // without description
    00120 
    00121     const std::map< G4String, G4MaterialPropertyVector*, std::less<G4String> >*
    00122     GetPropertiesMap() const { return &MPT; }
    00123     const std::map< G4String, G4double, std::less<G4String> >*
    00124     GetPropertiesCMap() const { return &MPTC; }
    00125     // Accessors required for persistency purposes


Follow the persistency clue::

    (chroma_env)delta:geant4.9.5.p01 blyth$ find . -name '*.cc' -exec grep -H GetPropertiesMap {} \;
    ./source/persistency/gdml/src/G4GDMLWriteMaterials.cc:                 std::less<G4String> >* pmap = ptable->GetPropertiesMap();
    (chroma_env)delta:geant4.9.5.p01 blyth$ 
    (chroma_env)delta:geant4.9.5.p01 blyth$ pwd
    /usr/local/env/chroma_env/src/geant4.9.5.p01



Map accessors not present in 4.9.2::

    [blyth@cms01 geant4.9.2.p01]$ vi source/materials/include/G4MaterialPropertiesTable.hh
    [blyth@cms01 geant4.9.2.p01]$ find . -name G4MaterialPropertiesTable.hh
    ./source/materials/include/G4MaterialPropertiesTable.hh
    ./include/G4MaterialPropertiesTable.hh

Maybe kludge by simply checking for existance of known hardcoded/configured keys::

    [blyth@cms01 dybgaudi]$ find . -name '*.cc' -exec grep -H GetProperty {} \; | perl -ne 'm,\"(\S*)\",&& print "$1\n" ' - | sort | uniq
    BACKSCATTERCONSTANT
    EFFICIENCY
    FASTCOMPONENT
    FASTTIMECONSTANT
    IMAGINARYRINDEX
    RAYLEIGH
    REALRINDEX
    REEMISSIONPROB
    REFLECTIVITY
    RESOLUTIONSCALE
    RINDEX
    SCINTILLATIONYIELD
    SLOWCOMPONENT
    SLOWTIMECONSTANT
    SPECULARLOBECONSTANT
    SPECULARSPIKECONSTANT
    YIELDRATIO




GDML persisting properties
---------------------------

**Patching seems more appropriate, as thats in the next G4 version anyhow.**

A G4DAE translation of the below with 4.9.2, will need to patch it to add public 
accessors to the maps.::

    (chroma_env)delta:geant4.9.5.p01 blyth$ vi source/persistency/gdml/src/G4GDMLWriteMaterials.cc

    208 void G4GDMLWriteMaterials::PropertyVectorWrite(const G4String& key,
    209                            const G4PhysicsOrderedFreeVector* const pvec)
    210 {
    211    const G4String matrixref = GenerateName(key, pvec);
    212    xercesc::DOMElement* matrixElement = NewElement("matrix");
    213    matrixElement->setAttributeNode(NewAttribute("name", matrixref));
    214    matrixElement->setAttributeNode(NewAttribute("coldim", "2"));
    215    std::ostringstream pvalues;
    216    for (size_t i=0; i<pvec->GetVectorLength(); i++)
    217    {
    218        if (i!=0)  { pvalues << " "; }
    219        pvalues << pvec->Energy(i) << " " << (*pvec)[i];
    220    }
    221    matrixElement->setAttributeNode(NewAttribute("values", pvalues.str()));
    222 
    223    defineElement->appendChild(matrixElement);
    224 }

    226 void G4GDMLWriteMaterials::PropertyWrite(xercesc::DOMElement* matElement,
    227                                          const G4Material* const mat)
    228 {
    229    xercesc::DOMElement* propElement;
    230    G4MaterialPropertiesTable* ptable = mat->GetMaterialPropertiesTable();
    231    const std::map< G4String, G4PhysicsOrderedFreeVector*,
    232                  std::less<G4String> >* pmap = ptable->GetPropertiesMap();
    233    const std::map< G4String, G4double,
    234                  std::less<G4String> >* cmap = ptable->GetPropertiesCMap();
    235    std::map< G4String, G4PhysicsOrderedFreeVector*,
    236                  std::less<G4String> >::const_iterator mpos;
    237    std::map< G4String, G4double,
    238                  std::less<G4String> >::const_iterator cpos;
    239    for (mpos=pmap->begin(); mpos!=pmap->end(); mpos++)
    240    {
    241       propElement = NewElement("property");
    242       propElement->setAttributeNode(NewAttribute("name", mpos->first));
    243       propElement->setAttributeNode(NewAttribute("ref",
    244                                     GenerateName(mpos->first, mpos->second)));
    245       if (mpos->second)
    246       {
    247          PropertyVectorWrite(mpos->first, mpos->second);
    248          matElement->appendChild(propElement);
    249       }
    250       else
    251       {
    252          G4String warn_message = "Null pointer for material property -"
    253                   + mpos->first + "- of material -" + mat->GetName() + "- !";
    254          G4Exception("G4GDMLWriteMaterials::PropertyWrite()", "NullPointer",
    255                      JustWarning, warn_message);
    256          continue;
    257       }
    258    }
    259    for (cpos=cmap->begin(); cpos!=cmap->end(); cpos++)
    260    {
    261       propElement = NewElement("property");
    262       propElement->setAttributeNode(NewAttribute("name", cpos->first));
    263       propElement->setAttributeNode(NewAttribute("ref", cpos->first));
    264       xercesc::DOMElement* constElement = NewElement("constant");
    265       constElement->setAttributeNode(NewAttribute("name", cpos->first));
    266       constElement->setAttributeNode(NewAttribute("value", cpos->second));
    267       defineElement->appendChild(constElement);
    268       matElement->appendChild(propElement);
    269    }
    270 }


Checking DAE Properties
------------------------

After :doc:`/geant4/nuwa/geant4_patch` to expose the properties and the adoption of GDML property 
writer into G4DAE succeed to includ properties in the DAE. But need some veracity checking::

     70373     <material id="__dd__Materials__Acrylic0xa7b6b48">
     70374       <instance_effect url="#__dd__Materials__Acrylic_fx_0xa7b6b48"/>
     70375       <extra>
     70376         <matrix coldim="2" name="ABSLENGTH0xa7b4d78" values="1.55e-06 8000 1.61e-06 8000 2.07e-06 8000 2.48e-06 8000 3.76e-06 8000 4.13e-06 8000 6.2e-06 0.008 1.033e-05 0.008 1.55e-05 0.008"/>
     70377         <property name="ABSLENGTH" ref="ABSLENGTH0xa7b4d78"/>
     70378         <matrix coldim="2" name="RAYLEIGH0xa7b4da8" values="1.55e-06 500000 1.7714e-06 300000 2.102e-06 170000 2.255e-06 100000 2.531e-06 62000 2.884e-06 42000 3.024e-06 30000 4.133e-06 7600 6.2e-06 850        1.033e-05 850 1.55e-05 850"/>
     70379         <property name="RAYLEIGH" ref="RAYLEIGH0xa7b4da8"/>
     70380         <matrix coldim="2" name="RINDEX0xa504f20" values="1.55e-06 1.4878 1.79505e-06 1.4895 2.10499e-06 1.4925 2.27077e-06 1.4946 2.55111e-06 1.4986 2.84498e-06 1.5022 3.06361e-06 1.5065 4.13281e-06 1.       5358 6.2e-06 1.6279 6.526e-06 1.627 6.889e-06 1.5359 7.294e-06 1.5635 7.75e-06 1.793 8.267e-06 1.7199 8.857e-06"/>
     70381         <property name="RINDEX" ref="RINDEX0xa504f20"/>
     70382       </extra>
     70383     </material>

Confirmed values truncation at 255 chars, maybe extend that buffer or move values to text content::

     70497     <material id="__dd__Materials__ESR0xa56f4b0">
     70498       <instance_effect url="#__dd__Materials__ESR_fx_0xa56f4b0"/>
     70499       <extra>
     70500         <matrix coldim="2" name="ABSLENGTH0xa8080f8" values="1.55e-06 0.001 1.63e-06 0.001 1.68e-06 0.001 1.72e-06 0.001 1.77e-06 0.001 1.82e-06 0.001 1.88e-06 0.001 1.94e-06 0.001 2e-06 0.001 2.07e-06        0.001 2.14e-06 0.001 2.21e-06 0.001 2.3e-06 0.001 2.38e-06 0.001 2.48e-06 0.001 2.58e-06 0.001 2.7e-06 0.001 2.82e"/>
     70501         <property name="ABSLENGTH" ref="ABSLENGTH0xa8080f8"/>
     70502       </extra>
     70503     </material>

What about constant properties ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~







Need to propagate surface properties too
------------------------------------------

Recent Reflectivity Change
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* http://dayabay.ihep.ac.cn/tracs/dybsvn/browser/dybgaudi/trunk/Detector/XmlDetDesc/DDDB/AdDetails/properties.xml

Change in surface properties has recently been discussed::

     This is caused by reflectivity of radial shield which was updated by Logan.
     The number is changed from 0.07 to about 0.04.

::

    g4pb:materials blyth$ grep opticalsurface g4_00.dae.2
        <opticalsurface finish="3" model="1" name="/dd/Geometry/PoolDetails/NearPoolSurfaces/NearPoolCoverSurface" type="0" value="1"/>
        <opticalsurface finish="3" model="1" name="/dd/Geometry/AdDetails/AdSurfacesAll/RSOilSurface" type="0" value="1"/>
        <opticalsurface finish="0" model="1" name="/dd/Geometry/AdDetails/AdSurfacesAll/ESRAirSurfaceTop" type="0" value="0"/>
        <opticalsurface finish="0" model="1" name="/dd/Geometry/AdDetails/AdSurfacesAll/ESRAirSurfaceBot" type="0" value="0"/>
        <opticalsurface finish="3" model="1" name="/dd/Geometry/AdDetails/AdSurfacesAll/SSTOilSurface" type="0" value="1"/>
        <opticalsurface finish="3" model="1" name="/dd/Geometry/AdDetails/AdSurfacesAll/AdCableTraySurface" type="0" value="1"/>
        <opticalsurface finish="3" model="1" name="/dd/Geometry/AdDetails/AdSurfacesNear/SSTWaterSurfaceNear1" type="0" value="1"/>
        <opticalsurface finish="3" model="1" name="/dd/Geometry/AdDetails/AdSurfacesNear/SSTWaterSurfaceNear2" type="0" value="1"/>
        <opticalsurface finish="3" model="1" name="/dd/Geometry/PoolDetails/PoolSurfacesAll/PmtMtTopRingSurface" type="0" value="1"/>
        <opticalsurface finish="3" model="1" name="/dd/Geometry/PoolDetails/PoolSurfacesAll/PmtMtBaseRingSurface" type="0" value="1"/>
        <opticalsurface finish="3" model="1" name="/dd/Geometry/PoolDetails/PoolSurfacesAll/PmtMtRib1Surface" type="0" value="1"/>
        <opticalsurface finish="3" model="1" name="/dd/Geometry/PoolDetails/PoolSurfacesAll/PmtMtRib2Surface" type="0" value="1"/>
        <opticalsurface finish="3" model="1" name="/dd/Geometry/PoolDetails/PoolSurfacesAll/PmtMtRib3Surface" type="0" value="1"/>
        <opticalsurface finish="3" model="1" name="/dd/Geometry/PoolDetails/PoolSurfacesAll/LegInIWSTubSurface" type="0" value="1"/>
        <opticalsurface finish="3" model="1" name="/dd/Geometry/PoolDetails/PoolSurfacesAll/TablePanelSurface" type="0" value="1"/>
        <opticalsurface finish="3" model="1" name="/dd/Geometry/PoolDetails/PoolSurfacesAll/SupportRib1Surface" type="0" value="1"/>
        <opticalsurface finish="3" model="1" name="/dd/Geometry/PoolDetails/PoolSurfacesAll/SupportRib5Surface" type="0" value="1"/>
        <opticalsurface finish="3" model="1" name="/dd/Geometry/PoolDetails/PoolSurfacesAll/SlopeRib1Surface" type="0" value="1"/>
        <opticalsurface finish="3" model="1" name="/dd/Geometry/PoolDetails/PoolSurfacesAll/SlopeRib5Surface" type="0" value="1"/>
        <opticalsurface finish="3" model="1" name="/dd/Geometry/PoolDetails/PoolSurfacesAll/ADVertiCableTraySurface" type="0" value="1"/>
        <opticalsurface finish="3" model="1" name="/dd/Geometry/PoolDetails/PoolSurfacesAll/ShortParCableTraySurface" type="0" value="1"/>
        <opticalsurface finish="3" model="1" name="/dd/Geometry/PoolDetails/NearPoolSurfaces/NearInnInPiperSurface" type="0" value="1"/>
        <opticalsurface finish="3" model="1" name="/dd/Geometry/PoolDetails/NearPoolSurfaces/NearInnOutPiperSurface" type="0" value="1"/>
        <opticalsurface finish="3" model="1" name="/dd/Geometry/PoolDetails/NearPoolSurfaces/NearIWSCurtainSurface" type="0" value="0.2"/>
        <opticalsurface finish="3" model="1" name="/dd/Geometry/PoolDetails/PoolSurfacesAll/LegInOWSTubSurface" type="0" value="1"/>
        <opticalsurface finish="3" model="1" name="/dd/Geometry/PoolDetails/NearPoolSurfaces/UnistrutRib6Surface" type="0" value="1"/>
        <opticalsurface finish="3" model="1" name="/dd/Geometry/PoolDetails/NearPoolSurfaces/UnistrutRib7Surface" type="0" value="1"/>
        <opticalsurface finish="3" model="1" name="/dd/Geometry/PoolDetails/PoolSurfacesAll/UnistrutRib3Surface" type="0" value="1"/>
        <opticalsurface finish="3" model="1" name="/dd/Geometry/PoolDetails/PoolSurfacesAll/UnistrutRib5Surface" type="0" value="1"/>
        <opticalsurface finish="3" model="1" name="/dd/Geometry/PoolDetails/PoolSurfacesAll/UnistrutRib4Surface" type="0" value="1"/>
        <opticalsurface finish="3" model="1" name="/dd/Geometry/PoolDetails/PoolSurfacesAll/UnistrutRib1Surface" type="0" value="1"/>
        <opticalsurface finish="3" model="1" name="/dd/Geometry/PoolDetails/PoolSurfacesAll/UnistrutRib2Surface" type="0" value="1"/>
        <opticalsurface finish="3" model="1" name="/dd/Geometry/PoolDetails/PoolSurfacesAll/UnistrutRib8Surface" type="0" value="1"/>
        <opticalsurface finish="3" model="1" name="/dd/Geometry/PoolDetails/PoolSurfacesAll/UnistrutRib9Surface" type="0" value="1"/>
        <opticalsurface finish="3" model="1" name="/dd/Geometry/PoolDetails/NearPoolSurfaces/TopShortCableTraySurface" type="0" value="1"/>
        <opticalsurface finish="3" model="1" name="/dd/Geometry/PoolDetails/PoolSurfacesAll/TopCornerCableTraySurface" type="0" value="1"/>
        <opticalsurface finish="3" model="1" name="/dd/Geometry/PoolDetails/PoolSurfacesAll/VertiCableTraySurface" type="0" value="1"/>
        <opticalsurface finish="3" model="1" name="/dd/Geometry/PoolDetails/NearPoolSurfaces/NearOutInPiperSurface" type="0" value="1"/>
        <opticalsurface finish="3" model="1" name="/dd/Geometry/PoolDetails/NearPoolSurfaces/NearOutOutPiperSurface" type="0" value="1"/>
        <opticalsurface finish="3" model="1" name="/dd/Geometry/PoolDetails/NearPoolSurfaces/NearOWSLinerSurface" type="0" value="0.2"/>
        <opticalsurface finish="3" model="1" name="/dd/Geometry/PoolDetails/PoolSurfacesAll/LegInDeadTubSurface" type="0" value="1"/>
        <opticalsurface finish="3" model="1" name="/dd/Geometry/PoolDetails/NearPoolSurfaces/NearDeadLinerSurface" type="0" value="0.2"/>
    g4pb:materials blyth$ 


Need to find new position inside extra, tacked under geometries is no good::

      1648         </extra>
      1649       </mesh>
      1650     </geometry>
      1651     <opticalsurface finish="3" model="1" name="/dd/Geometry/PoolDetails/NearPoolSurfaces/NearPoolCoverSurface" type="0" value="1"/>
      1652     <geometry id="RPCStrip0xb62fda0" name="RPCStrip0xb62fda0">
      1653       <mesh>
      1654         <source id="RPCStrip0xb62fda0-Pos">
      1655           <float_array count="24" id="RPCStrip0xb62fda0-Pos-array">
      1656                 -1040 -130 -1


::

    157517         <instance_node url="#__dd__Geometry__Sites__lvNearSiteRock0xb6131d8"/>
    157518         <extra>
    157519           <meta id="/dd/Structure/Sites/db-rock0xbeebd90">
    157520             <copyNo>1000</copyNo>
    157521             <ModuleName></ModuleName>
    157522           </meta>
    157523         </extra>
    157524       </node>
    157525     </node>
    157526     <skinsurface name="/dd/Geometry/PoolDetails/NearPoolSurfaces/NearPoolCoverSurface" surfaceproperty="/dd/Geometry/PoolDetails/NearPoolSurfaces/NearPoolCoverSurface">
    157527       <volumeref ref="/dd/Geometry/PoolDetails/lvNearTopCover0xb8a6408"/>
    157528     </skinsurface>
    157529     <skinsurface name="/dd/Geometry/AdDetails/AdSurfacesAll/RSOilSurface" surfaceproperty="/dd/Geometry/AdDetails/AdSurfacesAll/RSOilSurface">
    157530       <volumeref ref="/dd/Geometry/AdDetails/lvRadialShieldUnit0xb9b5ef0"/>
    157531     </skinsurface>
    157532     <skinsurface name="/dd/Geometry/AdDetails/AdSurfacesAll/AdCableTraySurface" surfaceproperty="/dd/Geometry/AdDetails/AdSurfacesAll/AdCableTraySurface">
    157533       <volumeref ref="/dd/Geometry/AdDetails/lvAdVertiCableTray0xba34d38"/>
    157534     </skinsurface>
    157535     <skinsurface name="/dd/Geometry/PoolDetails/PoolSurfacesAll/PmtMtTopRingSurface" surfaceproperty="/dd/Geometry/PoolDetails/PoolSurfacesAll/PmtMtTopRingSurface">
    157536       <volumeref ref="/dd/Geometry/PMT/lvPmtTopRing0xba4f460"/>
    157537     </skinsurface>



    157622     <skinsurface name="/dd/Geometry/PoolDetails/NearPoolSurfaces/NearOutOutPiperSurface" surfaceproperty="/dd/Geometry/PoolDetails/NearPoolSurfaces/NearOutOutPiperSurface">
    157623       <volumeref ref="/dd/Geometry/PoolDetails/lvOutOutWaterPipeNearTub0xbee65d0"/>
    157624     </skinsurface>
    157625     <skinsurface name="/dd/Geometry/PoolDetails/PoolSurfacesAll/LegInDeadTubSurface" surfaceproperty="/dd/Geometry/PoolDetails/PoolSurfacesAll/LegInDeadTubSurface">
    157626       <volumeref ref="/dd/Geometry/PoolDetails/lvLegInDeadTub0xbee6630"/>
    157627     </skinsurface>

    157628     <bordersurface name="/dd/Geometry/AdDetails/AdSurfacesAll/ESRAirSurfaceTop" surfaceproperty="/dd/Geometry/AdDetails/AdSurfacesAll/ESRAirSurfaceTop">
    157629       <physvolref ref="/dd/Geometry/AdDetails/lvTopReflector#pvTopRefGap0xb6d81c8"/>
    157630       <physvolref ref="/dd/Geometry/AdDetails/lvTopRefGap#pvTopESR0xb657cf0"/>
    157631     </bordersurface>
    157632     <bordersurface name="/dd/Geometry/AdDetails/AdSurfacesAll/ESRAirSurfaceBot" surfaceproperty="/dd/Geometry/AdDetails/AdSurfacesAll/ESRAirSurfaceBot">
    157633       <physvolref ref="/dd/Geometry/AdDetails/lvBotReflector#pvBotRefGap0xb57a378"/>
    157634       <physvolref ref="/dd/Geometry/AdDetails/lvBotRefGap#pvBotESR0xb95ad38"/>
    157635     </bordersurface>
    157636     <bordersurface name="/dd/Geometry/AdDetails/AdSurfacesAll/SSTOilSurface" surfaceproperty="/dd/Geometry/AdDetails/AdSurfacesAll/SSTOilSurface">
    157637       <physvolref ref="/dd/Geometry/AD/lvSST#pvOIL0xb579938"/>
    157638       <physvolref ref="/dd/Geometry/AD/lvADE#pvSST0xb6aff00"/>
    157639     </bordersurface>
    157640     <bordersurface name="/dd/Geometry/AdDetails/AdSurfacesNear/SSTWaterSurfaceNear1" surfaceproperty="/dd/Geometry/AdDetails/AdSurfacesNear/SSTWaterSurfaceNear1">
    157641       <physvolref ref="/dd/Geometry/Pool/lvNearPoolIWS#pvNearADE10xb5a95a8"/>
    157642       <physvolref ref="/dd/Geometry/AD/lvADE#pvSST0xb6aff00"/>
    157643     </bordersurface>
    157644     <bordersurface name="/dd/Geometry/AdDetails/AdSurfacesNear/SSTWaterSurfaceNear2" surfaceproperty="/dd/Geometry/AdDetails/AdSurfacesNear/SSTWaterSurfaceNear2">
    157645       <physvolref ref="/dd/Geometry/Pool/lvNearPoolIWS#pvNearADE20xb5ad858"/>
    157646       <physvolref ref="/dd/Geometry/AD/lvADE#pvSST0xb6aff00"/>
    157647     </bordersurface>
    157648     <bordersurface name="/dd/Geometry/PoolDetails/NearPoolSurfaces/NearIWSCurtainSurface" surfaceproperty="/dd/Geometry/PoolDetails/NearPoolSurfaces/NearIWSCurtainSurface">
    157649       <physvolref ref="/dd/Geometry/Pool/lvNearPoolCurtain#pvNearPoolIWS0xb914f38"/>
    157650       <physvolref ref="/dd/Geometry/Pool/lvNearPoolOWS#pvNearPoolCurtain0xb9a79d0"/>
    157651     </bordersurface>
    157652     <bordersurface name="/dd/Geometry/PoolDetails/NearPoolSurfaces/NearOWSLinerSurface" surfaceproperty="/dd/Geometry/PoolDetails/NearPoolSurfaces/NearOWSLinerSurface">
    157653       <physvolref ref="/dd/Geometry/Pool/lvNearPoolLiner#pvNearPoolOWS0xbe81770"/>
    157654       <physvolref ref="/dd/Geometry/Pool/lvNearPoolDead#pvNearPoolLiner0xb6772a0"/>
    157655     </bordersurface>
    157656     <bordersurface name="/dd/Geometry/PoolDetails/NearPoolSurfaces/NearDeadLinerSurface" surfaceproperty="/dd/Geometry/PoolDetails/NearPoolSurfaces/NearDeadLinerSurface">
    157657       <physvolref ref="/dd/Geometry/Sites/lvNearHallBot#pvNearPoolDead0xb56ff90"/>
    157658       <physvolref ref="/dd/Geometry/Pool/lvNearPoolDead#pvNearPoolLiner0xb6772a0"/>
    157659     </bordersurface>
    157660   </library_nodes>


properties too
~~~~~~~~~~~~~~~~

Added properties inside the opticalsurface element::

      1650     </geometry>
      1651     <opticalsurface finish="3" model="1" name="/dd/Geometry/PoolDetails/NearPoolSurfaces/NearPoolCoverSurface" type="0" value="1">
      1652       <matrix coldim="2" name="REFLECTIVITY0xadabaa8" values="1.5e-06 0 6.5e-06 0"/>
      1653       <property name="REFLECTIVITY" ref="REFLECTIVITY0xadabaa8"/>
      1654       <matrix coldim="2" name="RINDEX0xadabc88" values="1.5e-06 0 6.5e-06 0"/>
      1655       <property name="RINDEX" ref="RINDEX0xadabc88"/>
      1656     </opticalsurface>
      1657     <geometry id="RPCStrip0xa4c5e20" name="RPCStrip0xa4c5e20">
      1658       <mesh>
      1659         <source id="RPCStrip0xa4c5e20-Pos">
      1660           <float_array count="24" id="RPCStrip0xa4c5e20-Pos-array">
      1661                 -1040 -130 -1
      1662                 1040 -130 -1



how to position the elements inside DAE extra
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For material properties, its obvious what to do::

     70543   <library_materials>
     70544     <material id="__dd__Materials__PPE0xa543b68">
     70545       <instance_effect url="#__dd__Materials__PPE_fx_0xa543b68"/>
     70546       <extra>
     70547         <matrix coldim="2" name="ABSLENGTH0xa49fcb8" values="1.55e-06 0.001 6.2e-06 0.001 1.033e-05 0.001 1.55e-05 0.001"/>
     70548         <property name="ABSLENGTH" ref="ABSLENGTH0xa49fcb8"/>
     70549       </extra>
     70550     </material>

Not so obvious for bordersurface or skinsurface, which reference one or two volumes and an opticalsurface element.



Completeness ?
-----------------


::

    191   <!-- Radial shield with Oil -->
    192   <tabproperty name="RSOilReflectivity"
    193                type="REFLECTIVITY"
    194                xunit="eV"
    195                yunit=""
    196                xaxis="Energy"
    197                yaxis="Reflectivity">
    198     1.55 0.0393
    199     1.771 0.0393
    200     2.066 0.0394
    201     2.480 0.03975
    202     2.755 0.04045
    203     3.010 0.04135
    204     3.542 0.0432
    205     4.133 0.04655
    206     4.959 0.0538
    207     6.20 0.067
    208     10.33 0.114
    209     15.5 0.173
    210   </tabproperty>
    211 



