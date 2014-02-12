Material Properties
====================

Objective
----------

Persist material optical properties (some of which are wavelength dependent) 
into G4DAE COLLADA XML using extra tags, thus allowing access from pycollada 
and thence into Chroma.

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



