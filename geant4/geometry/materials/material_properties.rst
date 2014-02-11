Material Properties
====================

Objective
----------

Persist material optical properties (some of which are wavelength dependent) 
into G4DAE COLLADA XML using extra tags, thus allowing access from pycollada 
and thence into Chroma.

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



