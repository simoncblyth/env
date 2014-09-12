
DetSim DsPmtSensDet 
=====================

Overview
---------

#. `DsPmtSensDet::ProcessHits` hit formation has messy detector specific code, but not too extensive
   
   * expect GPU doable without extreme efforts
   * PMT identification is the most involved aspect 


Questions
~~~~~~~~~~

* At what juncture to merge GPU formed hits back into Geant4/NuWa/DetSim ?

  * maybe form hits in NewStage and fill the appropriate hit collection, 
    then there should be nothing else to do 

* where is the GiGa/Geant4/DetSim handoff happening ?  DsPullEvent

* where are SensDet identified ? 

  * DetDesc **sensdet** attribute on **logvol** elements, just yields two SD: `DsRpcSensDet` and `DsPmtSensDet`  



DsPmtSensDet
-------------

DsPmtSensDet::Initialize create HC for each (site,det) for each event
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
::

    195 void DsPmtSensDet::Initialize(G4HCofThisEvent* hce)
    196 {
    197     m_hc.clear();
    198 
    199     G4DhHitCollection* hc = new G4DhHitCollection(SensitiveDetectorName,collectionName[0]);
    200     m_hc[0] = hc;
    201     int hcid = G4SDManager::GetSDMpointer()->GetCollectionID(hc);
    202     hce->AddHitsCollection(hcid,hc);
    203 
    204     for (int isite=0; site_ids[isite] >= 0; ++isite) {
    205         for (int idet=0; detector_ids[idet] >= 0; ++idet) {
    206             DayaBay::Detector det(site_ids[isite],detector_ids[idet]);
    207 
    208             if (det.bogus()) continue;
    209 
    210             string name=det.detName();
    211             G4DhHitCollection* hc = new G4DhHitCollection(SensitiveDetectorName,name.c_str());
    212             short int id = det.siteDetPackedData();
    213             m_hc[id] = hc;
    214 
    215             int hcid = G4SDManager::GetSDMpointer()->GetCollectionID(hc);
    216             hce->AddHitsCollection(hcid,hc);
    217             debug() << "Add hit collection with hcid=" << hcid << ", cached ID="
    218                     << (void*)id
    219                     << " name= \"" << SensitiveDetectorName << "/" << name << "\""
    220                     << endreq;
    221         }
    222     }
    223 
    224     debug() << "DsPmtSensDet Initialize, made "
    225            << hce->GetNumberOfCollections() << " collections"
    226            << endreq;
    227    
    228 }





SensDet Identification
------------------------
::

    [blyth@belle7 DDDB]$ find . -name '*.xml' -exec grep -H Sens {} \;
    ./RPC/RPCStrip.xml:  <logvol name="lvRPCStrip" material="MixGas" sensdet="DsRpcSensDet">
    ./PMT/headon-pmt.xml:  <logvol name="lvHeadonPmtCathode" material="Bialkali" sensdet="DsPmtSensDet">
    ./PMT/hemi-pmt.xml:  <logvol name="lvPmtHemiCathode" material="Bialkali" sensdet="DsPmtSensDet">


`DDDB/PMT/headon-pmt.xml`::

     72   <!-- The Photo Cathode -->
     73   <logvol name="lvHeadonPmtCathode" material="Bialkali" sensdet="DsPmtSensDet">
     74     <tubs name="headon-pmt-cath"
     75           sizeZ="HeadonPmtCathodeThickness"
     76       outerRadius="HeadonPmtGlassRadius-HeadonPmtGlassWallThick"/>
     77   </logvol>

`DDDB/PMT/hemi-pmt.xml`::

    118   <!-- The Photo Cathode -->
    119   <!-- use if limit photocathode to a face on diameter gt 167mm. -->
    120   <logvol name="lvPmtHemiCathode" material="Bialkali" sensdet="DsPmtSensDet">
    121     <union name="pmt-hemi-cathode">
    122       <sphere name="pmt-hemi-cathode-face"
    123           outerRadius="PmtHemiFaceROCvac"
    124           innerRadius="PmtHemiFaceROCvac-PmtHemiCathodeThickness"
    125           deltaThetaAngle="PmtHemiFaceCathodeAngle"/>
    126       <sphere name="pmt-hemi-cathode-belly"
    127           outerRadius="PmtHemiBellyROCvac"
    128           innerRadius="PmtHemiBellyROCvac-PmtHemiCathodeThickness"
    129           startThetaAngle="PmtHemiBellyCathodeAngleStart"
    130           deltaThetaAngle="PmtHemiBellyCathodeAngleDelta"/>
    131       <posXYZ z="PmtHemiFaceOff-PmtHemiBellyOff"/>
    132     </union>
    133   </logvol>


Translation of detdesc into Geant4
-----------------------------------

::

    [blyth@belle7 lhcb]$ find . -name '*.cpp'  -exec grep -H sensdet {} \;
    ./Det/DetDescCnv/src/component/XmlLVolumeCnv.cpp:  sensdetString = xercesc::XMLString::transcode("sensdet");
    ./Det/DetDescCnv/src/component/XmlLVolumeCnv.cpp:  xercesc::XMLString::release((XMLCh**)&sensdetString);
    ./Det/DetDescCnv/src/component/XmlLVolumeCnv.cpp:  std::string sensDetName = dom2Std (element->getAttribute (sensdetString));

`NuWa-trunk/lhcb/Det/DetDescCnv/src/component/XmlLVolumeCnv.cpp`::

     405     // if there is a solid, creates a logical volume and stores the solid inside
     406     dataObj = new LVolume(volName,
     407                           solid,
     408                           materialName,
     409                           sensDetName,
     410                           magFieldName);


`NuWa-trunk/lhcb/Det/DetDesc/src/Lib/LVolume.cpp`::

     36 // ===========================================================================
     37 /*  constructor, pointer to ISolid* must be valid!, 
     38  *  overvise constructor throws LVolumeException!  
     39  *  @exception LVolumeException wrong paramaters value
     40  *  @param name         name of logical volume 
     41  *  @param Solid        pointer to ISolid object 
     42  *  @param material     name of the material 
     43  *  @param sensitivity  name of sensitive detector object (for simulation)
     44  *  @param magnetic     name of magnetic field object (for simulation)
     45  */
     46 // =========================================================================== 
     47 LVolume::LVolume
     48 ( const std::string& name        ,
     49   ISolid*            Solid       ,
     50   const std::string& material    ,
     51   const std::string& sensitivity ,
     52   const std::string& magnetic    )
     53   : LogVolBase     ( name        ,
     54                      sensitivity ,
     55                      magnetic    )
     56   , m_solid        ( Solid       )
     57   , m_materialName ( material    )
     58   , m_material     (    0        )
     59 {
     60   if( 0 == m_solid )
     61     { throw LogVolumeException("LVolume: ISolid* points to NULL ") ; }
     62 }



GiGa conversion of intermediary LVolume into G4LogicalVolume
--------------------------------------------------------------

::

    [blyth@belle7 GiGaCnv]$ find . -name '*.cpp' -exec grep -H sens {} \; 
    ./src/component/GiGaLAssemblyCnv.cpp:  /// sensitivity
    ./src/component/GiGaLAssemblyCnv.cpp:    { return Error("LAssembly could not be sensitive (now)"            ) ; }
    ./src/component/GiGaLVolumeCnv.cpp:  // sensitivity
    ./src/component/GiGaLVolumeCnv.cpp:      StatusCode sc = geoSvc()->sensitive( lv->sdName(), det );
    ./src/component/GiGaLVolumeCnv.cpp:      // set sensitive detector 
    ./src/component/GiGaLVolumeCnv.cpp:    // set sensitive detector 
    ./src/component/GiGaGeo.cpp:  // manually finalize all created sensitive detectors
    ./src/component/GiGaGeo.cpp:StatusCode   GiGaGeo::sensitive   
    ./src/component/GiGaGeo.cpp:  // inform Geant4 sensitive detector manager  
    ./src/component/GiGaGeo.cpp:StatusCode GiGaGeo::sensDet
    ./src/component/GiGaGeo.cpp:  Warning(" sensDet() is the obsolete method, use sensitive()!");
    ./src/component/GiGaGeo.cpp:  return sensitive( TypeNick , SD ) ;  
    ./src/component/GiGaGeo.cpp:      StatusCode sc = sensitive( m_budget , budget );
    [blyth@belle7 GiGaCnv]$ pwd
    /data1/env/local/dyb/NuWa-trunk/lhcb/Sim/GiGaCnv


`NuWa-trunk/lhcb/Sim/GiGaCnv/src/component/GiGaLVolumeCnv.cpp`::

    185   // sensitivity
    186   if( !lv->sdName().empty() ) {
    187     if( 0 == G4LV->GetSensitiveDetector() ) {
    188       IGiGaSensDet* det = 0 ;
    189       StatusCode sc = geoSvc()->sensitive( lv->sdName(), det );
    190       if( sc.isFailure() ) {
    191         return Error("updateRep:: Could no create SensDet ", sc );
    192       }
    193       if( 0 == det ) {
    194         return Error("updateRep:: Could no create SensDet ");
    195       }
    196       // set sensitive detector 
    197       G4LV->SetSensitiveDetector( det );
    198     } else {
    199       Warning( "SensDet is already defined to be '" +
    200                GiGaUtil::ObjTypeName( G4LV->GetSensitiveDetector() ) +"'");
    201     }
    202   }

`NuWa-trunk/lhcb/Sim/GiGaCnv/src/component/GiGaGeo.cpp`::

    751 //=============================================================================
    752 // Instantiate the Sensitive Detector Object 
    753 //=============================================================================
    754 StatusCode   GiGaGeo::sensitive
    755 ( const std::string& name  ,
    756   IGiGaSensDet*&     det   )
    757 {
    758   // reset the output value 
    759   det = 0 ;
    760   // locate the detector 
    761   det = tool( name , det , this );
    762   if( 0 == det )
    763     { return Error( "Could not locate Sensitive Detector='" + name + "'" ) ; }
    764   // inform Geant4 sensitive detector manager  
    765   if( m_SDs.end() == std::find( m_SDs.begin() , m_SDs.end  () , det ) )
    766     {
    767       G4SDManager* SDman = G4SDManager::GetSDMpointer();
    768       if( 0 == SDman ) { return Error( "Could not locate G4SDManager" ) ; }
    769       SDman -> AddNewDetector( det );
    770     }
    771   // keep local copy 
    772   m_SDs.push_back( det );
    773   ///
    774   return StatusCode::SUCCESS;
    775 };


`NuWa-trunk/lhcb/Sim/GiGa/GiGa/IGiGaSensDet.h`::

     22 class IGiGaSensDet: public virtual G4VSensitiveDetector,
     23                     public virtual IGiGaInterface
     24 {
     25 public:
     26 
     27   /** Retrieve the unique interface ID (static)
     28    *  @see IInterface
     29    */
     30   static const InterfaceID& interfaceID();
     31 
     32   /** Method for being a member of a GiGaSensDetSequence
     33    *  Implemented by base class, does not need reimplementation!
     34    */
     35   virtual bool processStep( G4Step* step, G4TouchableHistory* history ) = 0;
     36 
     37 protected:
     38 
     39   virtual ~IGiGaSensDet(); ///< virtual destructor 
     40   IGiGaSensDet() ;         ///< default constructor  
     41 
     42 };


::

     58 //=============================================================================
     59 // initialize the sensitive detector (Gaudi)
     60 //=============================================================================
     61 StatusCode GiGaSensDetBase::initialize()
     62 {
     63   StatusCode sc = GiGaBase::initialize() ;
     64   if( sc.isFailure() ) {
     65     return Error("Could not initialize base class GiGaBase");
     66   }
     67 
     68   // Correct the names!
     69   {
     70 
     71     std::string detname(name());
     72     std::string::size_type posdot = detname.find(".");
     73     detname.erase(0,posdot+1);
     74 
     75     std::string tmp( m_detPath + "/" + detname );
     76     std::string::size_type pos = tmp.find("//") ;
     77     while( std::string::npos != pos )
     78       { tmp.erase( pos , 1 ) ; pos = tmp.find("//") ; }
     79 
     80     // attention!!! direct usage of G4VSensitiveDetector members!!!! 
     81     pos = tmp.find_last_of('/') ;
     82     if( std::string::npos == pos )
     83       {
     84         G4VSensitiveDetector::SensitiveDetectorName = tmp ;  /// ATTENTION !!!
     85         G4VSensitiveDetector::thePathName           = "/" ;  /// ATTENTION !!! 
     86       }
     87     else
     88       {
     89         G4VSensitiveDetector::SensitiveDetectorName = tmp              ;
     90         G4VSensitiveDetector::SensitiveDetectorName.remove(0,pos+1)    ;
     91         G4VSensitiveDetector::thePathName           = tmp              ;
     92         G4VSensitiveDetector::thePathName.remove(pos+1,tmp.length()-1) ;
     93         if( '/' != G4VSensitiveDetector::thePathName[(unsigned int)(0)] )
     94           { G4VSensitiveDetector::thePathName.insert(0,"/"); }
     95       }
     96     ///
     97     G4VSensitiveDetector::fullPathName =
     98       G4VSensitiveDetector::thePathName +
     99       G4VSensitiveDetector::SensitiveDetectorName;
     ...   


Generalisable Identifier Heist ?
---------------------------------

* hmm, maybe can do something generalisable for SD by grabbing identifiers from Geant4 
  and persisting them into COLLADA export ?

  * are the identifiers there to be grabbed though ? 


`source/geometry/management/include/G4LogicalVolume.hh`::

    281     inline G4VSensitiveDetector* GetSensitiveDetector() const;
    282       // Gets current SensitiveDetector.
    283     inline void SetSensitiveDetector(G4VSensitiveDetector *pSDetector);
    284       // Sets SensitiveDetector (can be 0).



DsPmtSensDet
--------------

`NuWa-trunk/dybgaudi/Simulation/DetSim/src/DsPmtSensDet.h`::

     26 class DsPmtSensDet : public GiGaSensDetBase {
     27 public:
     28     DsPmtSensDet(const std::string& type,
     29                  const std::string& name,
     30                  const IInterface*  parent);
     31     virtual ~DsPmtSensDet();
     32 
     33     // G4VSensitiveDetector interface
     34     virtual void Initialize( G4HCofThisEvent* HCE ) ;
     35     virtual void EndOfEvent( G4HCofThisEvent* HCE ) ;
     36     virtual bool ProcessHits(G4Step* step,
     37                              G4TouchableHistory* history);
     38 
     39     // Tool interface
     40     virtual StatusCode initialize();
     41     virtual StatusCode finalize();
     42 
     43 private:
     44     /// Properties:
     45 
     46     /// CathodeLogicalVolumes : name of logical volumes in which this
     47     /// sensitive detector is operating.
     48     std::vector<std::string> m_cathodeLogVols;
     49 
     50     /// SensorStructures : names of paths in TDS in which to search
     51     /// for sensor detector elements using this sensitive detector.
     52     std::vector<std::string> m_sensorStructures;
     53 
     54     /// PackedIdParameterName : name of user paramater of the counted
     55     /// detector element which holds the packed, globally unique PMT
     56     /// ID.
     57     std::string m_idParameter;
     58 
     59     /// TouchableToDetelem : the ITouchableToDetectorElement to use to
     60     /// resolve sensor ID.
     61     std::string m_t2deName;
     62     ITouchableToDetectorElement* m_t2de;
     63 
     64     /// QEScale: Upward adjustment of DetSim efficiency to allow
     65     /// PMT-to-PMT efficiency variation in the electronics simulation.
     66     /// The value should be the inverse of the mean PMT efficiency
     67     /// applied in ElecSim.
     68     double m_qeScale;
     69 
     70     /// 
     71     bool m_ConvertWeightToEff;
     72 
     73     /// QEffParameterName : name of user parameter in the photo
     74     /// cathode volume that holds the quantum efficiency tabproperty.
     75     std::string m_qeffParamName;
     76 
     77     // Store hit in a hit collection
     78     void StoreHit(DayaBay::SimPmtHit* hit, int trackid);
     79 


DsPmtSensDet::DsPmtSensDet
----------------------------

::

     56 DsPmtSensDet::DsPmtSensDet(const std::string& type,
     57                            const std::string& name,
     58                            const IInterface*  parent)
     59     : G4VSensitiveDetector(name)
     60     , GiGaSensDetBase(type,name,parent)
     61     , m_t2de(0)
     62 {
     63     info() << "DsPmtSensDet (" << type << "/" << name << ") created" << endreq;
     64 
     65     declareProperty("CathodeLogicalVolume",
     66                     m_cathodeLogVols,
     67                     "Photo-Cathode logical volume to which this SD is attached.");
     68 
     69     declareProperty("TouchableToDetelem", m_t2deName = "TH2DE",
     70                     "The ITouchableToDetectorElement to use to resolve sensor.");
     71 
     72     declareProperty("SensorStructures",m_sensorStructures,
     73                     "TDS Paths in which to look for sensor detector elements"
     74                     " using this sensitive detector");
     75 
     76     declareProperty("PackedIdPropertyName",m_idParameter="PmtID",
     77                     "The name of the user property holding the PMT ID.");
     78 
     79     declareProperty("QEffParameterName",m_qeffParamName="EFFICIENCY",
     80                     "name of user parameter in the photo cathode volume that"
     81                     " holds the quantum efficiency tabproperty");
     82 
     83     declareProperty("QEScale",m_qeScale=1.0 / 0.9,
     84                     "Upward scaling of the quantum efficiency by inverse of mean PMT-to-PMT efficiency in electronics simulation.");
     85 
     86     declareProperty("ConvertWeightToEff", m_ConvertWeightToEff=false,
     87                     "Treat to the optical photon weight as to preliminary applied QE."
     88                     "Will affect only the primary photons (GtDiffuserBallTool, etc.).");
     89    
     90     m_cathodeLogVols.push_back("/dd/Geometry/PMT/lvPmtHemiCathode");
     91     m_cathodeLogVols.push_back("/dd/Geometry/PMT/lvHeadonPmtCathode");
     92 }


::

    [blyth@belle7 dybgaudi]$ find . -name '*.cc' -exec grep -H SensorStructures  {} \;
    ./Simulation/DetSim/src/DsPmtSensDet.cc:    declareProperty("SensorStructures",m_sensorStructures,
    ./Simulation/DetSim/src/DsRpcSensDet.cc:    declareProperty("SensorStructures",m_sensorStructures,




DsPmtSensDet::ProcessHits SimPmtHit formation from G4Step, stored into hit collections 
-----------------------------------------------------------------------------------------

`NuWa-trunk/dybgaudi/Simulation/DetSim/src/DsPmtSensDet.cc`::

    318 bool DsPmtSensDet::ProcessHits(G4Step* step,
    319                                G4TouchableHistory* /*history*/)
    320 {
    321     //if (!step) return false; just crash for now if not defined
    322 
    323     // Find out what detector we are in (ADx, IWS or OWS)
    324     G4StepPoint* preStepPoint = step->GetPreStepPoint();
    325 
    326     double energyDep = step->GetTotalEnergyDeposit();
    327 
    328     if (energyDep <= 0.0) {
    329         //debug() << "Hit energy too low: " << energyDep/CLHEP::eV << endreq;
    330         return false;
    331     }
    332 
    333     const G4TouchableHistory* hist =
    334         dynamic_cast<const G4TouchableHistory*>(preStepPoint->GetTouchable());
    335     if (!hist or !hist->GetHistoryDepth()) {
    336         error() << "ProcessHits: step has no or empty touchable history" << endreq;
    337         return false;
    338     }
    339 
    340     const DetectorElement* de = this->SensDetElem(*hist);
    341     if (!de) return false;
    342 
    343     // wangzhe QE calculation starts here.
    344     int pmtid = this->SensDetId(*de);
    345     DayaBay::Detector detector(pmtid);
    346     G4Track* track = step->GetTrack();
    347     double weight = track->GetWeight();
    ...
    459     DayaBay::SimPmtHit* sphit = new DayaBay::SimPmtHit();
    460 
    461     // base hit
    462 
    463     // Time since event created
    464     sphit->setHitTime(preStepPoint->GetGlobalTime());
    465 
    466     //#include "G4NavigationHistory.hh"
    467 
    468     const G4AffineTransform& trans = hist->GetHistory()->GetTopTransform();
    469     const G4ThreeVector& global_pos = preStepPoint->GetPosition();
    470     G4ThreeVector pos = trans.TransformPoint(global_pos);
    471     sphit->setLocalPos(pos);
    472     sphit->setSensDetId(pmtid);
    ...
    505     int trackid = track->GetTrackID();
    506     this->StoreHit(sphit,trackid);
    507     debug() << "Stored photon " << trackid << " weight " << weight << " pmtid " << (void*)pmtid << " wavelength(nm) " << wavelength/CLHEP::nm << e    ndreq;
    508     return true;
    509 }
    ...
    511 void DsPmtSensDet::StoreHit(DayaBay::SimPmtHit* hit, int trackid)
    512 {
    513     int did = hit->sensDetId();
    514     DayaBay::Detector det(did);
    515     short int sdid = det.siteDetPackedData();
    516 
    517     G4DhHitCollection* hc = m_hc[sdid];
    518     if (!hc) {
    519         warning() << "Got hit with no hit collection.  ID = " << (void*)did
    520                   << " which is detector: \"" << DayaBay::Detector(did).detName()
    521                   << "\". Storing to the " << collectionName[0] << " collection"
    522                   << endreq;
    523         sdid = 0;
    524         hc = m_hc[sdid];
    525     }
    526 
    527 #if 1
    528     verbose() << "Storing hit PMT: " << (void*)did
    529               << " from " << DayaBay::Detector(did).detName()
    530               << " in hc #"<<  sdid << " = "
    531               << hit->hitTime()/CLHEP::ns << "[ns] "
    532               << hit->localPos()/CLHEP::cm << "[cm] "
    533               << hit->wavelength()/CLHEP::nm << "[nm]"
    534               << endreq;
    535 #endif
    536 
    537     hc->insert(new G4DhHit(hit,trackid));
    538 }



GiGaSensDetBase
---------------

`NuWa-trunk/lhcb/Sim/GiGa/GiGa/GiGaSensDetBase.h`::

     22 class GiGaSensDetBase: virtual public IGiGaSensDet ,
     23                        public GiGaBase
     24 {
     ..
     60   /** Method for being a member of a GiGaSensDetSequence
     61    *  Implemented by base class, does not need reimplementation!
     62    */
     63   virtual bool processStep( G4Step* step,
     64                             G4TouchableHistory* history );
     ..
     75   bool                m_active  ;  ///< Active Flag
     76   std::string         m_detPath ;
     77 };

`NuWa-trunk/lhcb/Sim/GiGa/GiGa/IGiGaSensDet.h`::

     22 class IGiGaSensDet: public virtual G4VSensitiveDetector,
     23                     public virtual IGiGaInterface
     24 {
     25 public:
     ..
     35   virtual bool processStep( G4Step* step, G4TouchableHistory* history ) = 0;
     36 


`NuWa-trunk/lhcb/Sim/GiGa/src/Lib/GiGaSensDetBase.cpp`::

    152 // ============================================================================
    153 bool GiGaSensDetBase::processStep( G4Step* step,
    154                                    G4TouchableHistory* history ) {
    155   // delegate to ProcessHits
    156   return ProcessHits( step, history );
    157 
    158 }


G4VSensitiveDetector
-----------------------

`geant4.10.00.p01/source/digits_hits/detector/include/G4VSensitiveDetector.hh`::

     50 class G4VSensitiveDetector
     51 {
     52 
     53   public: // with description
     54       G4VSensitiveDetector(G4String name);
     ..
     68   public: // with description
     69       virtual void Initialize(G4HCofThisEvent*);
     70       virtual void EndOfEvent(G4HCofThisEvent*);
     71       //  These two methods are invoked at the begining and at the end of each
     72       // event. The hits collection(s) created by this sensitive detector must
     73       // be set to the G4HCofThisEvent object at one of these two methods.
     74       virtual void clear();
     75       //  This method is invoked if the event abortion is occured. Hits collections
     76       // created but not beibg set to G4HCofThisEvent at the event should be deleted.
     77       // Collection(s) which have already set to G4HCofThisEvent will be deleted 
     78       // automatically.
     ..
     84   protected: // with description
     85       virtual G4bool ProcessHits(G4Step*aStep,G4TouchableHistory*ROhist) = 0;
     86       //  The user MUST implement this method for generating hit(s) using the 
     87       // information of G4Step object. Note that the volume and the position
     88       // information is kept in PreStepPoint of G4Step.
     89       //  Be aware that this method is a protected method and it sill be invoked 
     90       // by Hit() method of Base class after Readout geometry associated to the
     91       // sensitive detector is handled.
     92       //  "ROhist" will be given only is a Readout geometry is defined to this
     93       // sensitive detector. The G4TouchableHistory object of the tracking geometry
     94       // is stored in the PreStepPoint object of G4Step.
     95       virtual G4int GetCollectionID(G4int i);
     96       //  This is a utility method which returns the hits collection ID of the
     97       // "i"-th collection. "i" is the order (starting with zero) of the collection
     98       // whose name is stored to the collectionName protected vector.
     99       G4CollectionNameVector collectionName;
     00       //  This protected name vector must be filled at the constructor of the user's
     01       // concrete class for registering the name(s) of hits collection(s) being
     02       // created by this particular sensitive detector.



GDB Session Probe G4SDManager
------------------------------

::

    (gdb) p G4SDManager::GetSDMpointer()
    [Switching to Thread -1208218944 (LWP 11466)]
    $1 = (G4SDManager *) 0xb24d3d0
    Current language:  auto; currently c++
    (gdb) p G4SDManager::GetSDMpointer()->ListTree()
    $2 = void

stdout from the ListTree::
 
    /
    /DsRpcSensDet   *** Active 
    /DsPmtSensDet   *** Active 



::

    (gdb) p G4SDManager::GetSDMpointer()->GetCollectionCapacity()
    Cannot evaluate function -- may be inlined
    (gdb) p G4SDManager::GetSDMpointer()->GetHCTable()
    Couldn't find method G4SDManager::GetHCTable
    (gdb) p G4SDManager::GetSDMpointer()->GetHCtable()
    $3 = (G4HCtable *) 0xb330d38
    (gdb) p G4SDManager::GetSDMpointer()->GetHCtable()->entries()
    $4 = 23


::

    (gdb) p G4SDManager::GetSDMpointer()->GetHCtable()->GetSDname(0)
    Cannot evaluate function -- may be inlined
    (gdb) p G4SDManager::GetSDMpointer()->GetHCtable()->GetHCname(0)
    Cannot evaluate function -- may be inlined
    (gdb) p G4SDManager::GetSDMpointer()->GetHCtable()->GetHCname(1)
    Cannot evaluate function -- may be inlined

    (gdb) p G4SDManager::GetSDMpointer()->GetHCtable()->SDlist[4]
    $11 = (const G4String &) @0xb267230: {<std::basic_string<char,std::char_traits<char>,std::allocator<char> >> = {static npos = 4294967295, 
        _M_dataplus = {<std::allocator<char>> = {<__gnu_cxx::new_allocator<char>> = {<No data fields>}, <No data fields>}, _M_p = 0xb36b6d4 "DsPmtSensDet"}}, <No data fields>}
    (gdb) p G4SDManager::GetSDMpointer()->GetHCtable()->SDlist[5]
    $12 = (const G4String &) @0xb267234: {<std::basic_string<char,std::char_traits<char>,std::allocator<char> >> = {static npos = 4294967295, 
        _M_dataplus = {<std::allocator<char>> = {<__gnu_cxx::new_allocator<char>> = {<No data fields>}, <No data fields>}, _M_p = 0xb36b6d4 "DsPmtSensDet"}}, <No data fields>}
    (gdb) p G4SDManager::GetSDMpointer()->GetHCtable()->SDlist[6]
    $13 = (const G4String &) @0xb267238: {<std::basic_string<char,std::char_traits<char>,std::allocator<char> >> = {static npos = 4294967295, 
        _M_dataplus = {<std::allocator<char>> = {<__gnu_cxx::new_allocator<char>> = {<No data fields>}, <No data fields>}, _M_p = 0xb36b6d4 "DsPmtSensDet"}}, <No data fields>}
    (gdb) p G4SDManager::GetSDMpointer()->GetHCtable()->SDlist[7]
    $14 = (const G4String &) @0xb26723c: {<std::basic_string<char,std::char_traits<char>,std::allocator<char> >> = {static npos = 4294967295, 
        _M_dataplus = {<std::allocator<char>> = {<__gnu_cxx::new_allocator<char>> = {<No data fields>}, <No data fields>}, _M_p = 0xb36b6d4 "DsPmtSensDet"}}, <No data fields>}
    (gdb) p G4SDManager::GetSDMpointer()->GetHCtable()->HClist[7]
    $15 = (const G4String &) @0xb147254: {<std::basic_string<char,std::char_traits<char>,std::allocator<char> >> = {static npos = 4294967295, 
        _M_dataplus = {<std::allocator<char>> = {<__gnu_cxx::new_allocator<char>> = {<No data fields>}, <No data fields>}, _M_p = 0xb32aea4 "DayaBayAD3"}}, <No data fields>}
    (gdb) p G4SDManager::GetSDMpointer()->GetHCtable()->HClist[8]
    $16 = (const G4String &) @0xb147258: {<std::basic_string<char,std::char_traits<char>,std::allocator<char> >> = {static npos = 4294967295, 
        _M_dataplus = {<std::allocator<char>> = {<__gnu_cxx::new_allocator<char>> = {<No data fields>}, <No data fields>}, _M_p = 0xb32aec4 "DayaBayAD4"}}, <No data fields>}
    (gdb) p G4SDManager::GetSDMpointer()->GetHCtable()->HClist[9]
    $17 = (const G4String &) @0xb14725c: {<std::basic_string<char,std::char_traits<char>,std::allocator<char> >> = {static npos = 4294967295, 
        _M_dataplus = {<std::allocator<char>> = {<__gnu_cxx::new_allocator<char>> = {<No data fields>}, <No data fields>}, _M_p = 0xb32aee4 "DayaBayIWS"}}, <No data fields>}
    (gdb) p G4SDManager::GetSDMpointer()->GetHCtable()->HClist[10]
    $18 = (const G4String &) @0xb147260: {<std::basic_string<char,std::char_traits<char>,std::allocator<char> >> = {static npos = 4294967295, 
        _M_dataplus = {<std::allocator<char>> = {<__gnu_cxx::new_allocator<char>> = {<No data fields>}, <No data fields>}, _M_p = 0xb32af04 "DayaBayOWS"}}, <No data fields>}
    (gdb) 



`source/digits_hits/detector/include/G4SDManager.hh`::

     50 class G4SDManager
     51 {
     52   public: // with description
     53       static G4SDManager* GetSDMpointer();
     54       // Returns the pointer to the singleton object.
     55   public:
     56       static G4SDManager* GetSDMpointerIfExist();
     57 
     58   protected:
     59       G4SDManager();
     60 
     61   public:
     62       ~G4SDManager();
     63 
     64   public: // with description
     65       void AddNewDetector(G4VSensitiveDetector*aSD);
     66       //  Registors the user's sensitive detector. This method must be invoked
     67       // when the user construct his/her sensitive detector.
     68       void Activate(G4String dName, G4bool activeFlag);
     69       //  Activate/inactivate the registered sensitive detector. For the inactivated
     70       // detectors, hits collections will not be stored to the G4HCofThisEvent object.
     71       G4int GetCollectionID(G4String colName);
     72       G4int GetCollectionID(G4VHitsCollection * aHC);
     73       //  These two methods return the ID number of the sensitive detector.
     74 
     75   public:
     76       G4VSensitiveDetector* FindSensitiveDetector(G4String dName, G4bool warning = true);
     77       G4HCofThisEvent* PrepareNewEvent();
     78       void TerminateCurrentEvent(G4HCofThisEvent* HCE);
     79       void AddNewCollection(G4String SDname,G4String DCname);
     80 
     81 
     82   private:
     83       static G4ThreadLocal G4SDManager * fSDManager;
     84       G4SDStructure * treeTop;
     85       G4int verboseLevel;
     86       G4HCtable* HCtable;
     87       G4SDmessenger* theMessenger;
     88 


`source/digits_hits/detector/src/G4SDManager.cc`::

     67 void G4SDManager::AddNewDetector(G4VSensitiveDetector*aSD)
     68 {
     69   G4int numberOfCollections = aSD->GetNumberOfCollections();
     70   G4String pathName = aSD->GetPathName();
     71   if( pathName(0) != '/' ) pathName.prepend("/");
     72   if( pathName(pathName.length()-1) != '/' ) pathName += "/";
     73   treeTop->AddNewDetector(aSD,pathName);
     74   if(numberOfCollections<1) return;
     75   for(G4int i=0;i<numberOfCollections;i++)
     76   {
     77     G4String SDname = aSD->GetName();
     78     G4String DCname = aSD->GetCollectionName(i);
     79     AddNewCollection(SDname,DCname);
     80   }
     81   if( verboseLevel > 0 )
     82   {
     83     G4cout << "New sensitive detector <" << aSD->GetName()
     84          << "> is registored at " << pathName << G4endl;
     85   }
     86 }


::

     47 class G4SDStructure
     48 {
     49   public:
     50       G4SDStructure(G4String aPath);
     51       ~G4SDStructure();
     52 
     53       G4int operator==(const G4SDStructure &right) const;
     54 
     55       void AddNewDetector(G4VSensitiveDetector*aSD, G4String treeStructure);
     56       void Activate(G4String aName, G4bool sensitiveFlag);
     57       void Initialize(G4HCofThisEvent*HCE);
     58       void Terminate(G4HCofThisEvent*HCE);
     59       G4VSensitiveDetector* FindSensitiveDetector(G4String aName, G4bool warning = true);
     60       G4VSensitiveDetector* GetSD(G4String aName);
     61       void ListTree();
     62 
     63   private:
     64       G4SDStructure* FindSubDirectory(G4String subD);
     65       G4String ExtractDirName(G4String aPath);
     66       void RemoveSD(G4VSensitiveDetector*);
     67 
     68   private:
     69       std::vector<G4SDStructure*> structure;
     70       std::vector<G4VSensitiveDetector*> detector;
     71       G4String pathName;
     72       G4String dirName;
     73       G4int verboseLevel;



Hmm nothing there, killed all photons ? Might be true, but empty implementations anyhow::

    (gdb) p G4SDManager::GetSDMpointer()->treeTop->detector[0]->PrintAll()
    $24 = void
    (gdb) p G4SDManager::GetSDMpointer()->treeTop->detector[1]->PrintAll()
    $25 = void


    (gdb) p G4SDManager::GetSDMpointer()->treeTop->detector[1]->collectionName.size()
    $30 = 19

    (gdb) p G4SDManager::GetSDMpointer()->treeTop->detector[1]->collectionName[0]    
    $31 = (const G4String &) @0xb3ce458: {<std::basic_string<char,std::char_traits<char>,std::allocator<char> >> = {static npos = 4294967295, 
        _M_dataplus = {<std::allocator<char>> = {<__gnu_cxx::new_allocator<char>> = {<No data fields>}, <No data fields>}, _M_p = 0xb248be4 "unknown"}}, <No data fields>}
    (gdb) p G4SDManager::GetSDMpointer()->treeTop->detector[1]->collectionName[1]
    $32 = (const G4String &) @0xb3ce45c: {<std::basic_string<char,std::char_traits<char>,std::allocator<char> >> = {static npos = 4294967295, 
        _M_dataplus = {<std::allocator<char>> = {<__gnu_cxx::new_allocator<char>> = {<No data fields>}, <No data fields>}, _M_p = 0xb248bfc "DayaBayAD1"}}, <No data fields>}
    (gdb) p G4SDManager::GetSDMpointer()->treeTop->detector[1]->collectionName[2]
    $33 = (const G4String &) @0xb3ce460: {<std::basic_string<char,std::char_traits<char>,std::allocator<char> >> = {static npos = 4294967295, 
        _M_dataplus = {<std::allocator<char>> = {<__gnu_cxx::new_allocator<char>> = {<No data fields>}, <No data fields>}, _M_p = 0xb32ae84 "DayaBayAD2"}}, <No data fields>}
    (gdb) p G4SDManager::GetSDMpointer()->treeTop->detector[1]->collectionName[18]
    $34 = (const G4String &) @0xb3ce4a0: {<std::basic_string<char,std::char_traits<char>,std::allocator<char> >> = {static npos = 4294967295, 
        _M_dataplus = {<std::allocator<char>> = {<__gnu_cxx::new_allocator<char>> = {<No data fields>}, <No data fields>}, _M_p = 0xb3ce4ec "FarOWS"}}, <No data fields>}
    (gdb) 



`source/digits_hits/detector/src/G4HCtable.cc`::

     37 G4int G4HCtable::Registor(G4String SDname,G4String HCname)
     38 {
     39   for(size_t i=0;i<HClist.size();i++)
     40   { if(HClist[i]==HCname && SDlist[i]==SDname) return -1; }
     41   HClist.push_back(HCname);
     42   SDlist.push_back(SDname);
     43   return HClist.size();
     44 }
     45 
     46 G4int G4HCtable::GetCollectionID(G4String HCname) const
     //
     //   Collection list index of:
     //
     //        HCname          "DayaBayIWS" 
     //        SDname/HCname   "DsPmtSensDet/DayaBayAD4"    
     //
     47 {
     48   G4int i = -1;
     49   if(HCname.index("/")==std::string::npos) // HCname only
     50   {
     51     for(size_t j=0;j<HClist.size();j++)
     52     {
     53       if(HClist[j]==HCname)
     54       {
     55         if(i>=0) return -2;
     56         i = j;
     57       }
     58     }
     59   }
     60   else
     61   {
     62     for(size_t j=0;j<HClist.size();j++)
     63     {
     64       G4String tgt = SDlist[j];
     65       tgt += "/";
     66       tgt += HClist[j];
     67       if(tgt==HCname)
     68       {
     69         if(i>=0) return -2;
     70         i = j;
     71       }
     72     }
     73   }
     74   return i;
     75 }





