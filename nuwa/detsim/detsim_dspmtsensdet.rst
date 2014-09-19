
DetSim DsPmtSensDet 
=====================

Overview
---------

#. `DsPmtSensDet::ProcessHits` hit formation has messy detector specific code, but not too extensive
   
   * expect GPU doable without extreme efforts
   * PMT identification is the most involved aspect, did this within idmap

Questions
~~~~~~~~~~

* At what juncture to merge GPU formed hits back into Geant4/NuWa/DetSim ?

  * maybe form hits in NewStage and fill the appropriate hit collection, 
    then there should be nothing else to do 

* where is the GiGa/Geant4/DetSim handoff happening ?  DsPullEvent

* where are SensDet identified ? 

  * DetDesc **sensdet** attribute on **logvol** elements, just yields two SD: `DsRpcSensDet` and `DsPmtSensDet`  

* GiGa takes the detdesc geometry and converts into Geant4, 
  where exactly does that happen


Testing GPU Hit Formation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

#. running `csa.sh` takes too long to initialize to be a suitable way to test, 
   need to get sending dummy ChromaPhotonList working again, and then enhance it to 
   send real CPL sourced from a .root file 

   * see `czrt-` 


First Try
~~~~~~~~~~~

#. currently now QE, test without first 

::

    [blyth@belle7 dybgaudi]$ svn ci -m "minor: trying Geant4/Chroma integration via effectively forming hits on GPU and placing batches of them into hit collections in StackAction  " 
    Sending        Simulation/DetSimChroma/cmt/requirements
    Sending        Simulation/DetSimChroma/src/DsChromaStackAction.cc
    Sending        Simulation/DetSimChroma/src/DsChromaStackAction.h
    Sending        Utilities/Chroma/Chroma/ChromaPhotonList.hh
    Sending        Utilities/Chroma/cmt/requirements
    Sending        Utilities/Chroma/src/ChromaPhotonList.cc
    Transmitting file data ......
    Committed revision 23251.


Where does G4 call ProcessHits ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Via the header implemented `Hit` method of `G4VSensitiveDetector`


`geant4.10.00.p01/source/tracking/src/G4SteppingManager.cc`::

    116 G4StepStatus G4SteppingManager::Stepping()
    117 //////////////////////////////////////////
    118 {
    ...
    217 //-------
    218 // Finale
    219 //-------
    220 
    221 // Update 'TrackLength' and remeber the Step length of the current Step
    222    fTrack->AddTrackLength(fStep->GetStepLength());
    223    fPreviousStepSize = fStep->GetStepLength();
    224    fStep->SetTrack(fTrack);
    225 #ifdef G4VERBOSE
    226                          // !!!!! Verbose
    227 
    228            if(verboseLevel>0) fVerbose->StepInfo();
    229 #endif
    230 // Send G4Step information to Hit/Dig if the volume is sensitive
    231    fCurrentVolume = fStep->GetPreStepPoint()->GetPhysicalVolume();
    232    StepControlFlag =  fStep->GetControlFlag();
    233    if( fCurrentVolume != 0 && StepControlFlag != AvoidHitInvocation) {
    234       fSensitive = fStep->GetPreStepPoint()->
    235                                    GetSensitiveDetector();
    236       if( fSensitive != 0 ) {
    237         fSensitive->Hit(fStep);
    238       }
    239    }


G4VSensitiveDetector
~~~~~~~~~~~~~~~~~~~~~


`geant4.10.00.p01/source/digits_hits/detector/include/G4VSensitiveDetector.hh`::

    113   public: // with description
    114       inline G4bool Hit(G4Step*aStep)
    115       {
    116         G4TouchableHistory* ROhis = 0;
    117         if(!isActive()) return false;
    118         if(filter)
    119         { if(!(filter->Accept(aStep))) return false; }
    120         if(ROgeometry)
    121         { if(!(ROgeometry->CheckROVolume(aStep,ROhis))) return false; }
    122         return ProcessHits(aStep,ROhis);
    123       }
    124       //  This is the public method invoked by G4SteppingManager for generating
    125       // hit(s). The actual user's implementation for generating hit(s) must be
    126       // implemented in GenerateHits() virtual protected method. This method
    127       // MUST NOT be overrided.


`geant4.10.00.p01/source/digits_hits/detector/src/G4VSensitiveDetector.cc`::

    100 G4int G4VSensitiveDetector::GetCollectionID(G4int i)
    101 {
    102    return G4SDManager::GetSDMpointer()->GetCollectionID(SensitiveDetectorName+"/"+collectionName[i]);
    103 }
    104 
    105 //----- following methoods are abstract methods to be
    106 //----- implemented in the concrete classes
    107 
    108 void G4VSensitiveDetector::Initialize(G4HCofThisEvent*)
    109 {
    110 }
    111 
    112 void G4VSensitiveDetector::EndOfEvent(G4HCofThisEvent*)
    113 {
    114 }





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


DetDesc SensDet Identification
--------------------------------
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



How does gaudi go from sensdet attribute value DsPmtSensDet to instance
-------------------------------------------------------------------------


Looks like a GiGa hack::

    [blyth@belle7 lhcb]$ find . -name '*.cpp' -exec grep -H SensitiveDetectorName {} \;
    ./Sim/GiGa/src/Lib/GiGaSensDetBase.cpp:        G4VSensitiveDetector::SensitiveDetectorName = tmp ;  /// ATTENTION !!!
    ./Sim/GiGa/src/Lib/GiGaSensDetBase.cpp:        G4VSensitiveDetector::SensitiveDetectorName = tmp              ;
    ./Sim/GiGa/src/Lib/GiGaSensDetBase.cpp:        G4VSensitiveDetector::SensitiveDetectorName.remove(0,pos+1)    ;
    ./Sim/GiGa/src/Lib/GiGaSensDetBase.cpp:      G4VSensitiveDetector::SensitiveDetectorName; 




Mimic DsPmtSensDet within ChromaStackAction ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Need to pass along the solid index from the GPU, so can do a lookup into a volume store.  
Not needed for the pmtid, but do need to get the transform.
NB this is more than just a global position, the preStepPoint knows
which place its at in the geometry tree

::

    const G4TouchableHistory* hist = 
        dynamic_cast<const G4TouchableHistory*>(preStepPoint->GetTouchable());

    if (!hist or !hist->GetHistoryDepth()) {
        error() << "ProcessHits: step has no or empty touchable history" << endreq;
        return false;


::

    068 class G4StepPoint
     69 ///////////////// 
     70 {
    ...
    195    G4TouchableHandle fpTouchable;
    196       //  Touchable Handle  


`geant4.10.00.p01/source/track/include/G4StepPoint.icc`::

    140 inline
    141  const G4VTouchable* G4StepPoint::GetTouchable() const
    142  { return fpTouchable(); }
    143 



Can I use DsPmtSensDet methods from StackAction ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#. Service lookup only provides access to the minimal standard `IGiGaSensDet` interface, 
#. seems not: 

   * so need to operate at Geant4 level and duplicate 
     some of what DsPmtSensDet does to access the same 
     hit collections 

::

    [blyth@belle7 lhcb]$ find . -name '*.cpp'  -exec grep -H sensdet {} \;
    ./Det/DetDescCnv/src/component/XmlLVolumeCnv.cpp:  sensdetString = xercesc::XMLString::transcode("sensdet");
    ./Det/DetDescCnv/src/component/XmlLVolumeCnv.cpp:  xercesc::XMLString::release((XMLCh**)&sensdetString);
    ./Det/DetDescCnv/src/component/XmlLVolumeCnv.cpp:  std::string sensDetName = dom2Std (element->getAttribute (sensdetString));


LVolume attribute
~~~~~~~~~~~~~~~~~~

`NuWa-trunk/lhcb/Det/DetDescCnv/src/component/XmlLVolumeCnv.cpp`::

     405     // if there is a solid, creates a logical volume and stores the solid inside
     406     dataObj = new LVolume(volName,
     407                           solid,
     408                           materialName,
     409                           sensDetName,
     410                           magFieldName);


`NuWa-trunk/lhcb/Det/DetDesc/DetDesc/LVolume.h`::

     20 class LVolume: public LogVolBase
     21 {
     22   /// friend factory for instantiation 
     23   friend class DataObjectFactory<LVolume>;
     24 
     25 public:
     26 
     27   /** constructor, pointer to ISolid* must be valid!, 
     28    *  overvise constructor throws LogVolumeException!  
     29    *  @exception LVolumeException for wrong parameters set
     30    *  @param name         name of logical volume 
     31    *  @param Solid        pointer to ISolid object 
     32    *  @param material     name of the material 
     33    *  @param sensitivity  name of sensitive detector object (for simulation)
     34    *  @param magnetic     name of magnetic field object (for simulation)
     35    */
     36   LVolume( const std::string& name             ,
     37            ISolid*            Solid            ,
     38            const std::string& material         ,
     39            const std::string& sensitivity = "" ,
     40            const std::string& magnetic    = "" );
     41 
     42   /// destructor 
     43   virtual ~LVolume();


`NuWa-trunk/lhcb/Det/DetDesc/DetDesc/LogVolBase.h`::

    035 class LogVolBase:
     36   public virtual ILVolume   ,
     37   public         ValidDataObject
     38 {
     39 
     40 protected:
     41 
     42   /** constructor
     43    *  @exception LVolumeException wrong paramaters value
     44    *  @param name name of logical volume 
     45    *  @param sensitivity  name of sensitive detector object (for simulation)
     46    *  @param magnetic  nam eof magnetic field object (for simulation)
     47    */
     48   LogVolBase( const std::string& name        = "" ,
     49               const std::string& sensitivity = "" ,
     50               const std::string& magnetic    = "" );

    192   /** name of sensitive "detector" - needed for simulation 
    193    *  @see ILVolume 
    194    *  @return name of sensitive "detector"
    195    */
    196   inline virtual const std::string& sdName () const { return m_sdName; } ;



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


Where is top volume setup ?
-----------------------------

Annoyingly difficult to searchable API

`NuWa-trunk/lhcb/Sim/GiGa/src/component/GiGa.h`::

    215   /** set new world wolume 
    216    *               implementation of IGiGaSetUpSvc abstract interface 
    217    *
    218    *  NB: errors are reported through exception thrown 
    219    * 
    220    *  @param  world  pointer to  new world volume   
    221    *  @return self-reference ot IGiGaSetUpSvc interface 
    222    */
    223   virtual IGiGaSetUpSvc&
    224   operator << ( G4VPhysicalVolume             * world         ) ;


`NuWa-trunk/lhcb/Sim/GiGa/GiGa/IGiGaSetUpSvc.h`::

     49 class IGiGaSetUpSvc : virtual public IService
     50 {
     ..
     75 
     76   /** set new world wolume 
     77    * 
     78    *  @param  world  pointer to  new world volume   
     79    *  @return self-reference ot IGiGaSetUpSvc interface 
     80    */
     81   virtual IGiGaSetUpSvc& operator << ( G4VPhysicalVolume             * ) = 0 ;
      

`NuWa-trunk/lhcb/Sim/GiGa/src/component/GiGaIGiGaSetUpSvc.cpp`::

    085 // ============================================================================
    086 /** set new world wolume 
    087  *               implementation of IGiGaSetUpSvc abstract interface 
    088  *
    089  *  NB: errors are reported through exception thrown 
    090  * 
    091  *  @param  obj    pointer to  new world volume   
    092  *  @return self-reference ot IGiGaSetUpSvc interface 
    093  */
    094 // ============================================================================
    095 IGiGaSetUpSvc& GiGa::operator << ( G4VPhysicalVolume             * obj )
    096 {
    097   try
    098     {
    099       StatusCode sc = StatusCode::SUCCESS;
    100       if( 0 == runMgr  () ) { sc = retrieveRunManager()       ; }
    101       if( sc.isFailure () ) { Exception("Unable to create IGiGaRunManager!");}
    102       sc = runMgr()->declare( obj ) ;
    103       if( sc.isFailure () ) { Exception("Unable to declare" +
    104                                         GiGaUtil::ObjTypeName( obj ) ); }
    105     }
    106   catch ( const GaudiException& Excpt )
    107     { Exception( "operator<<(G4VPhysicalVolume*)" , Excpt ) ; }
    108   catch ( const std::exception& Excpt )
    109     { Exception( "operator<<(G4VPhysicalVolume*)" , Excpt ) ; }
    110   catch(...)
    111     { Exception( "operator<<(G4VPhysicalVolume*)"         ) ; }
    112   ///
    113   return *this;
    114 };


GiGaRunManager also handles geometry too
-------------------------------------------

Good for understanding GiGa action and source of breakpoints

* `NuWa-trunk/lhcb/Sim/GiGa/src/component/GiGaRunManager.cpp` 



`NuWa-trunk/lhcb/Sim/GiGa/src/component/GiGaRunManager.h`::

    047 class GiGaRunManager: public  virtual IGiGaRunManager  ,
    048                       public  virtual  GiGaBase        ,
    049                       private virtual G4RunManager
    050 {
    ...
    075   /** declare the top level ("world") physical volume 
    076    *  @see IGiGaRunManager 
    077    *  @param obj pointer  to top level ("world") physical volume  
    078    *  @return  status code 
    079    */
    080   virtual StatusCode declare( G4VPhysicalVolume              * obj ) ;
    081 
    ...
    269 private:
    270 
    271   bool                       m_krn_st          ;
    272   bool                       m_run_st          ;
    273   bool                       m_pre_st          ;
    274   bool                       m_pro_st          ;
    275   bool                       m_uis_st          ;
    276 
    277   G4VPhysicalVolume*         m_rootGeo         ;
    278   IGiGaGeoSrc*               m_geoSrc          ;
    279   G4UIsession*               m_g4UIsession     ;
    280 
    281   bool                       m_delDetConstr    ;
    282   bool                       m_delPrimGen      ;
    283   bool                       m_delPhysList     ;


`NuWa-trunk/lhcb/Sim/GiGa/src/component/GiGaRunManagerG4RM.cpp`::

     57 void GiGaRunManager::InitializeGeometry()
     58 {
     59   /// get root of geometry tree 
     60   G4VPhysicalVolume* root = 0;
     61   if      ( 0 != m_rootGeo                  )
     62     {
     63       Print(" Already converted geometry will be used!");
     64       root = m_rootGeo ;
     65     }
     66   else if ( 0 != geoSrc()                  )
     67     {
     68       Print(" Geometry will be extracted from " +
     69             GiGaUtil::ObjTypeName( geoSrc() ) );
     70       root = geoSrc()->world ();
     71     }
     72   else if ( 0 != G4RunManager::userDetector )
     73     {
     74       Print(" Geometry will be constructed using " +
     75             GiGaUtil::ObjTypeName( G4RunManager::userDetector ) );
     76       root = G4RunManager::userDetector->Construct() ;
     77     }
     78   else
     79     { Error(" There are NO known sources of Geometry information!"); }
     80   //
     81   if( 0 == root )
     82     { Exception("InitializeGeometry: NO 'geometry sources' abvailable");}
     83   ///  
     84   DefineWorldVolume( root ) ;
     85   G4RunManager::geometryInitialized = true;
     86 };





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
  Are the identifiers there to be grabbed though ?

  * Nope, the PMTID live as UserParam associated with DETDESC DetectorElement, it 
    seems these param are not propagated down into the Geant4 representation  


`source/geometry/management/include/G4LogicalVolume.hh`::

    281     inline G4VSensitiveDetector* GetSensitiveDetector() const;
    282       // Gets current SensitiveDetector.
    283     inline void SetSensitiveDetector(G4VSensitiveDetector *pSDetector);
    284       // Sets SensitiveDetector (can be 0).

    Dayabay has only two SensDet for Pmt and Rpc 


How to persist the PMTID in COLLADA
--------------------------------------

#. hmm adopt something like `G4GDMLAuxMapType` for G4DAE Export ? 

`geant4.10.00.p01/examples/extended/persistency/gdml/G04/gdml_det.cc`::


    103    // Example how to retrieve Auxiliary Information for sensitive detector
    104    //
    105    const G4GDMLAuxMapType* auxmap = parser.GetAuxMap();
    ...
    124    // The same as above, but now we are looking for
    125    // sensitive detectors setting them for the volumes
    126 
    127    for(G4GDMLAuxMapType::const_iterator iter=auxmap->begin();
    128        iter!=auxmap->end(); iter++)
    129    {
    130      G4cout << "Volume " << ((*iter).first)->GetName()
    131             << " has the following list of auxiliary information: "
    132             << G4endl << G4endl;
    133      for (G4GDMLAuxListType::const_iterator vit=(*iter).second.begin();
    134           vit!=(*iter).second.end();vit++)
    135      {
    136        if ((*vit).type=="SensDet")
    137        {
    138          G4cout << "Attaching sensitive detector " << (*vit).value
    139                 << " to volume " << ((*iter).first)->GetName()
    140                 <<  G4endl << G4endl;
       


GiGa Manual
------------

* http://lhcb-comp.web.cern.ch/lhcb-comp/Frameworks/Gaudi/Documents/GiGa.pdf


Section 3.2.3 Conversion of Geometry Objects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Geometry description in DETDESC package is made through 3 types of identifiable
objects `LVolume`, `DetectorElement` and `Surface`. 
The simplified class diagrams for 3 corresponding Converter classes `GiGaLVolumeCnv`, `GiGaDetectorElementCnv`
and `GiGaSurfaceCnv` are shown on figure 3.2. Call-backs from geometry Converters
to `IGiGaGeomCnvSvc` interface are explicitly indicated.

These classes are converted into GEANT4 classes `G4LogicalVolume`, `G4PVPlacement`,
`G4LogicalSkinSurface` and `G4LogicalBorderSurface`.

naming convention
^^^^^^^^^^^^^^^^^^^^

Logical volume (of type `G4LogicalVolume`) in GEANT4 get its name from `name()` 
method from `ILVolume` interface, which is the full address of 
logical volume in transient store, e.g. `/dd/Geometry/LHCb/lvLHCb`.

Situation with naming of physical volumes (of type `G4PVPlacement`) is a little
bit more complicated. Physical volume gets the name of the form 
`<MotherLVName>#PVname` 
if it is created during conversion of its mother logical volume 
or `FullPathForDetectorElement` if it corresponds to detector element, 
which is converted in a separate way without conversion of higher
level detector elements.

Surfaces (of types `G4LogicalSkinSurface` and `G4LogicalBorderSurface`) get their
name from `fullpath()` method of Surface class, e.g. `/dd/Geometry/Rich1/MirrorSurface`. 
The corresponding `G4OpticalSurface` class gets the same name.


GiGa geometry configuration
----------------------------

The global magnetic field is the property of GiGaGeomCnvSvc and could be configured through e.g job options technique:

::

    /// ...
    /// declare constant magnetic field as global field 
    GiGaGeomCnvSvc.WorldMagneticField = "GiGaMagFieldUniform/Uniform"; /// confiugure magnetic field
    Uniform.Bx = 0.0;
    Uniform.By = 10.0;
    Uniform.Bz = 10.0;
    /// ...


Top Down Trace from nuwa.py `-G/--detector`
---------------------------------------------

::

    [blyth@belle7 DybPython]$ grep detector *.py
    cmdline.py:    parser.add_argument("-G", "--detector",default= "",
    Control.py:        if self.opts.detector:
    Control.py:            XmlDetDesc.Configure(self.opts.detector)
    Control.py:                + self.opts.detector + " is loaded."
    DybPythonAlg.py:        detectorId = inputHeaders[0].context().GetDetId()
    DybPythonAlg.py:            # Extend time/detector range if needed
    DybPythonAlg.py:            if detectorId != DetectorId.kAll and detectorId != inputDetId:
    DybPythonAlg.py:                detectorId = DetectorId.kAll
    gaudiutil.py:            dec = "%2d %2d %2d %2d %d" % (pp.site(), pp.detectorId(), pp.inwardFacing(), pp.wallNumber(), pp.wallSpot())
    [blyth@belle7 DybPython]$ 


`NuWa-trunk/dybgaudi/Detector/XmlDetDesc/python/XmlDetDesc/__init__.py`::

     36         from XmlTools.XmlToolsConf import XmlCnvSvc, XmlParserSvc
     37         xmlcnv = XmlCnvSvc()
     38         xmlcnv.AllowGenericConversion = True
     39         xmlparser = XmlParserSvc()
     40 
     41         from Gaudi.Configuration import ApplicationMgr, DetectorPersistencySvc, DetectorDataSvc
     42 
     43         app = ApplicationMgr()
     44         app.ExtSvc += [ xmlcnv , xmlparser ]
     45 
     46         detper = DetectorPersistencySvc()
     47         detper.CnvServices.append(xmlcnv)
     48 
     49         detdat = DetectorDataSvc()
     50         detdat.UsePersistency = True
     51         detdat.DetDbRootName  = "dd"
     52         detdat.DetStorageType = 7
     53         detdat.DetDbLocation  = xmlfile



DetectorDataSvc
----------------

::

    [blyth@belle7 lhcb]$ find . -name '*.cpp' -exec grep -l DetectorDataSvc {} \;
    ./Tools/XmlTools/src/component/XmlParserSvc.cpp
    ./Sim/GiGaCnv/src/Lib/GiGaCnvBase.cpp
    ./Sim/GiGaCnv/src/Lib/GiGaCnvSvcBase.cpp
    ./Sim/GiGaCnv/src/component/GiGaGeo.cpp
    ./Det/DetDescSvc/src/EventClockSvc.cpp
    ./Det/DetDescSvc/src/PreloadGeometryTool.cpp
    ./Det/DetDescSvc/src/UpdateManagerSvc.cpp
    ./Det/DetDescSvc/src/TransportSvc.cpp
    ./Det/DetDescSvc/src/DetElemFinder.cpp
    ./Det/DetDesc/src/Lib/Services.cpp
    ./Vis/OnXSvc/src/OnXSvc.cpp


    [blyth@belle7 lhcb]$ cd ../gaudi
    [blyth@belle7 gaudi]$ find . -name '*.cpp' -exec grep -l DetectorDataSvc {} \;
    ./GaudiSvc/src/ApplicationMgr/ApplicationMgr.cpp
    ./GaudiSvc/src/DetectorDataSvc/DetDataSvc.cpp
    ./GaudiAlg/src/lib/GaudiTool.cpp
    ./GaudiKernel/src/Lib/Algorithm.cpp
    ./GaudiExamples/src/Properties/PropertyAlg.cpp


`NuWa-trunk/gaudi/GaudiSvc/src/DetectorDataSvc/DetDataSvc.cpp` looks to be lazy::

    207 /// Standard Constructor
    208 DetDataSvc::DetDataSvc(const std::string& name,ISvcLocator* svc) :
    209   DataSvc(name,svc), m_eventTime(0)  {
    210   declareProperty("DetStorageType",  m_detStorageType = XML_StorageType );
    211   declareProperty("DetDbLocation",   m_detDbLocation  = "empty" );
    212   declareProperty("DetDbRootName",   m_detDbRootName  = "dd" );
    213   declareProperty("UsePersistency",  m_usePersistency = false );
    214   declareProperty("PersistencySvc",  m_persistencySvcName = "DetectorPersistencySvc" );
    215   m_rootName = "/dd";
    216   m_rootCLID = CLID_Catalog;
    217   m_addrCreator = 0;
    218 }


* https://lhcb-comp.web.cern.ch/lhcb-comp/Frameworks/Gaudi/Gaudi_v9/GUG/Output/GUG_DetDescription.html


GiGaGeo : hunting control of DetDesc to Geant4 conversion
-----------------------------------------------------------

`NuWa-trunk/dybgaudi/Simulation/DetSim/python/DetSim/Default.py`::

     75         from GiGa.GiGaConf import GiGa
     76         giga = GiGa()
     77         giga.PhysicsList = physics_list
     78 
     79         # Start empty step action sequence to hold historian/unobserver
     80         from GaussTools.GaussToolsConf import GiGaStepActionSequence
     81         sa = GiGaStepActionSequence('GiGa.GiGaStepActionSequence')
     82         giga.SteppingAction = sa
     83 
     84         self.giga = giga
     85 
     86         # Tell GiGa the size of the world.
     87         # Set default world material to be vacuum to speed propagation of
     88         # particles in regions of little interest.
     89         from GiGaCnv.GiGaCnvConf import GiGaGeo
     90         giga_geom = GiGaGeo()
     91         giga_geom.XsizeOfWorldVolume = 2.4*units.kilometer
     92         giga_geom.YsizeOfWorldVolume = 2.4*units.kilometer
     93         giga_geom.ZsizeOfWorldVolume = 2.4*units.kilometer
     94         giga_geom.WorldMaterial = "/dd/Materials/Vacuum"
     95         self.gigageo = giga_geom
     96 
     97         # Set up for telling GiGa what geometry to use, but don't
     98         # actually set that.
     ..
     ..         set below GiGaInputStream section creating self.giga_items 
     ..
     13
     14         # Make sequencer alg to run all this stuff as subalgs
     15         from GaudiAlg.GaudiAlgConf import GaudiSequencer
     16         giga_sequence = GaudiSequencer()
     17         giga_sequence.Members = [ self.giga_items ]
     18         self.giga_sequence=giga_sequence
     19         if use_push_algs:
     20             # DetSim's algs
     21             from DetSim.DetSimConf import DsPushKine, DsPullEvent
     22             self.detsim_push_kine = DsPushKine()
     23             self.detsim_pull_event = DsPullEvent()
     24             giga_sequence.Members += [self.detsim_push_kine,
     25                                       self.detsim_pull_event]
     26             pass
     27 
     28         if not use_sim_subseq:
     29             from Gaudi.Configuration import ApplicationMgr
     30             theApp = ApplicationMgr()
     31             theApp.TopAlg.append(giga_sequence)



GiGaInputStream
-------------------

config
~~~~~~~~~

`NuWa-trunk/dybgaudi/Simulation/DetSim/python/DetSim/Default.py`::

     99         from GaussTools.GaussToolsConf import GiGaInputStream
     00         giga_items = GiGaInputStream()
     01         giga_items.ExecuteOnce = True
     02         giga_items.ConversionSvcName = "GiGaGeo"
     03         giga_items.DataProviderSvcName = "DetectorDataSvc"
     04         giga_items.StreamItems = [ ]
     05         site = site.lower()
     06         if "far" in site:
     07             giga_items.StreamItems += self.giga_far_items
     08         if "dayabay" in site:
     09             giga_items.StreamItems += self.giga_dayabay_items
     10         if "lingao" in site:
     11             giga_items.StreamItems += self.giga_lingao_items
     12         self.giga_items = giga_items

 
GiGaInputStream::execute
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Load objects (top level `/dd/Structure` paths) and apply GiGaGeo conversion 

#. add handful of top level `/dd/Structure` path for the simulated site to PreLoad list
#. DataSvc preload from DetDesc XML
#. run the `LoadObject` for each
#. `m_cnvSvc->createRep`
 
`NuWa-trunk/lhcb/Sim/GaussTools/src/Components/GiGaInputStream.cpp`::

     47 StatusCode GiGaInputStream::execute()
     48 {
     49   ///
     50   if( !m_execute ) { return StatusCode::SUCCESS; }
     51   ///
     52   MsgStream log( msgSvc() , name() );
     53   log << MSG::VERBOSE << " execute:: start" << endreq;
     54   ///
     55   if( m_executeOnce      ) { m_execute  = false; }
     56   ///
     57   /// preload data 
     58   Items::const_iterator item = m_items.begin() ;
     59   while( item != m_items.end() )
     60     { m_dataSvc->addPreLoadItem( *item++ ); }
     61   m_dataSvc->preLoad();
     62   ///
     63   StatusCode status = StatusCode::SUCCESS;
     64   m_dataSelector.clear();
     65   item = m_items.begin() ;
     66   while( item != m_items.end() && status.isSuccess() )
     67     { status = LoadObject( *item++, &m_dataSelector) ; }
     68   ///  
     69   if( status.isFailure() )
     70     { return Error("Execute::Could not load Object="+item->path(), status); }
     71   /// create the representation 
     72   for( IDataSelector::iterator obj1 = m_dataSelector.begin() ;
     73        m_dataSelector.end() != obj1 ; ++obj1 )
     74     {
     75       IOpaqueAddress* Address = 0 ;
     76       status = m_cnvSvc->createRep( *obj1 , Address ) ;
     77       if( status.isFailure() )
     78         { return Error(" Error in creation of representation!"); }
     79       // update the registry
     80       (*obj1)->registry()->setAddress( Address );
     81     }
     82   /// create the references 
     83   for( IDataSelector::iterator obj2 = m_dataSelector.begin() ;
     84        m_dataSelector.end() != obj2 ; ++obj2 )
     85     {
     86       status = m_cnvSvc->
     87         fillRepRefs( (*obj2)->registry()->address(),  *obj2 ) ;
     88       if( status.isFailure() )
     89         { return Error(" Error in creation of references!"); }
     90     }
     91   ///
     92   if( status.isFailure() )
     93     { return Error("Execute::Could not convert the IDataSelector*", status);}
     94   ///
     95   m_dataSelector.clear();
     96   ///
     97   log << MSG::VERBOSE << "Execute::end" << endreq;
     98   ///
     99   return status;
     00   ///
     01 };



`NuWa-trunk/lhcb/Sim/GiGaCnv/src/component/GiGaGeo.h`::

     29 /** @class GiGaGeo GiGaGeo.h
     30  *  
     31  *  Conversion service for transforming Gaudi detector 
     32  *  and geometry description into Geant4 geometry and 
     33  *  detector description 
     34  *  
     35  *  @author  Vanya Belyaev
     36  *  @author  Gonzalo Gracia
     37  *  @author  Sajan Easo, Gloria Corti
     38  *  @date    2000-08-07, Last modified: 2007-07-10
     39  */
     40 
     41 class GiGaGeo : public virtual  IGiGaGeomCnvSvc,
     42                 public           GiGaCnvSvcBase {
     43 

::

    144   /** Retrieve the pointer to top-level "world" volume,
    145    *  @see IGiGaGeo 
    146    *  needed for Geant4 - root for the whole Geant4 geometry tree 
    147    *  @see class IGiGaGeoSrc 
    148    *  @return pointer to constructed(converted) geometry tree 
    149    */
    150   virtual G4VPhysicalVolume* world();




`NuWa-trunk/lhcb/Sim/GiGaCnv/src/component/GiGaGeo.cpp`::

     79 //=============================================================================
     80 // Standard constructor, initializes variables
     81 //=============================================================================
     82 GiGaGeo::GiGaGeo( const std::string& serviceName,
     83                   ISvcLocator* serviceLocator )
     84   : GiGaCnvSvcBase( serviceName, serviceLocator, GiGaGeom_StorageType )
     85   , m_worldPV( 0 )
     86   , m_worldMagField( "" )   // check below for double properties 
     87   , m_SDs()
     88   , m_MFs()
     89   , m_FMs()
     90 {
     91 
     92   setNameOfDataProviderSvc("DetectorDataSvc");
     93 
     94   declareProperty( "WorldPhysicalVolumeName", m_worldNamePV = "Universe" );
     95   declareProperty( "WorldLogicalVolumeName",  m_worldNameLV = "World" );
     96   declareProperty( "WorldMaterial",   m_worldMaterial = "/dd/Materials/Air");
     97 
     98   declareProperty( "XsizeOfWorldVolume" , m_worldX = 50. * m );
     99   declareProperty( "YsizeOfWorldVolume" , m_worldY = 50. * m );
     00   declareProperty( "ZsizeOfWorldVolume" , m_worldZ = 50. * m );
     01 
     02   declareProperty( "GlobalSensitivity" , m_budget = "");
     03   // Probably obsolete: need to check if WorldMagneticField can be removed
     04   declareProperty( "WorldMagneticField", m_worldMagField );
     05   declareProperty( "FieldManager"      , m_worldMagField );
     06 
     07   declareProperty( "ClearStores", m_clearStores = true );
     08 
     09   declareProperty ( "UseAlignment",      m_useAlignment = false ) ;
     10   declareProperty ( "AlignAllDetectors", m_alignAll = false );
     11   m_alignDets.clear();
     12   declareProperty ( "AlignDetectors",    m_alignDets );
     13 
     14 };











DsPmtSensDet
--------------





`NuWa-trunk/lhcb/Sim/GiGa/GiGa/GiGaSensDetBase.h`::

     22 class GiGaSensDetBase: virtual public IGiGaSensDet ,
     23                        public GiGaBase
     24 {
     25 
     26 public:
     27 
     28   /** standard constructor   
     29    *  @see GiGaBase 
     30    *  @see AlgTool 
     31    *  @param type type of the object (?)
     32    *  @param name name of the object
     33    *  @param parent  pointer to parent object
     34    */
     35   GiGaSensDetBase ( const std::string& type   ,
     36                     const std::string& name   ,
     37                     const IInterface*  parent );



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


TotalEnergyDeposit
--------------------

`NuWa-trunk/dybgaudi/Simulation/DetSim/src`::

    [blyth@belle7 src]$ grep TotalEnergyDeposit *.cc
    DsG4Scintillation.cc:// necessary information resides in aStep.GetTotalEnergyDeposit()
    DsG4Scintillation.cc:    G4double TotalEnergyDeposit = aStep.GetTotalEnergyDeposit();
    DsG4Scintillation.cc:      G4cout << " TotalEnergyDeposit " << TotalEnergyDeposit 
    DsG4Scintillation.cc:    if (TotalEnergyDeposit <= 0.0 && !flagReemission) {
    DsG4Scintillation.cc:        G4double dE = TotalEnergyDeposit;
    DsG4Scintillation.cc:        G4double QuenchedTotalEnergyDeposit 
    DsG4Scintillation.cc:            = TotalEnergyDeposit/(1+birk1*delta+birk2*delta*delta);
    DsG4Scintillation.cc:        G4double MeanNumberOfPhotons= ScintillationYield * QuenchedTotalEnergyDeposit;
    DsPmtModel.cc:    fastStep.ProposeTotalEnergyDeposited(energy);
    DsPmtSensDet.cc:    double energyDep = step->GetTotalEnergyDeposit();
    DsRpcModel.cc:    fastStep.ProposeTotalEnergyDeposited(energy);
    DsRpcSensDet.cc:    double energyDep = step->GetTotalEnergyDeposit();


`NuWa-trunk/dybgaudi/Simulation/DetSim/src/DsPmtModel.cc`::

     61 void DsPmtModel::DoIt(const G4FastTrack& fastTrack, G4FastStep& fastStep)
     62 {
     63     const G4Track* track = fastTrack.GetPrimaryTrack();
     64     double energy = track->GetKineticEnergy();
     65 
     66     fastStep.ProposeTrackStatus(fStopAndKill);
     67     fastStep.ProposePrimaryTrackPathLength(0.0);
     68     fastStep.ProposeTotalEnergyDeposited(energy);
     69 }



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




DE to PMTID
-------------

`DetectorElement*->PmtId integer` 

* http://dayabay.ihep.ac.cn/tracs/dybsvn/browser/dybgaudi/trunk/Detector/XmlDetDescChecks/src/DeDumpAlg.cc


COLLADA ID
-----------

* COLLADA ID can be reproduced by 

  * full Geant4 geometry traverse accessing logical/physical names 
    and duplicating the `daenode.py` de-duping technique and encoding 
    for XML identifier 
  * this will by necessity visit all the PV in the traverse order 



COLLADA id to DetectorElement*
-----------------------------------------------------------------------


`NuWa-trunk/dybgaudi/Simulation/G4DataHelpers/src/components/TH2DE.h`::

     19 class TH2DE : public GaudiTool, virtual ITouchableToDetectorElement
     20 {
     21 public:
     22     TH2DE(const std::string& type,
     23          const std::string& name,
     24          const IInterface* parent);
     25     virtual ~TH2DE();
     26 
     27     virtual StatusCode GetBestDetectorElement(const G4TouchableHistory* inHistory,
     28                                               const std::vector<std::string>& /*ignored*/,
     29                                               const IDetectorElement* &outElement,
     30                                               int& outCompatibility);
     31 
     32     virtual StatusCode G4VolumeToDetDesc(const G4VPhysicalVolume* inVol,
     33                                          const IPVolume* &outVol);
     34 
     35     virtual StatusCode ClearCache();



`NuWa-trunk/Simulation/G4DataHelpers/src/components/TouchableToDetectorElementFast.h`::


     13 class TouchableToDetectorElementFast : public GaudiTool, virtual ITouchableToDetectorElement
     14 {
     15   public:
     16   TouchableToDetectorElementFast(const std::string& type,
     17                              const std::string& name,
     18                             const IInterface* parent);
     19   virtual ~TouchableToDetectorElementFast() {};
     20 
     21   /// Do the conversion.
     22   virtual StatusCode GetBestDetectorElement(
     23                           const G4TouchableHistory* inHistory,  // The current particle location
     24                           const std::vector<std::string>& inPath,// Name(s) of specific detElements, or paths to be searched
     25                           const IDetectorElement* &outElement,  // output: The best element (may be zero!) 
     26                           int& outCompatibility );              // output: the goodness of the match (lower is better, <=0 means no match.)
     27 
     28   /// Utility to do a simpler conversion: find the IPVolume from a G4PhysicalVolume.
     29   virtual StatusCode G4VolumeToDetDesc( const G4VPhysicalVolume* inVol,     ///< Input G4 volume
     30                                         const IPVolume* &outVol             ///< Output DetDesc volume
     31                                        );
     32 
     33   /// Clear caches in case of geometry change.
     34  virtual StatusCode ClearCache();
     35 
     36  private:
     37 
     38    // Things to store in the cache:
     39    struct Relation {
     40      Relation() : mLogVol(0), mPhysVol(0), mRpath(0) {};
     41      bool isNull() const { return mLogVol==0; };
     42      const ILVolume*          mLogVol; // The supporting volume
     43      const IPVolume*          mPhysVol;
     44      ILVolume::ReplicaPath    mRpath;  // Empty if it is not under the mWorldElement.
     45    };
     46 
     47    // The Cache:
     48    const IDetectorElement* mWorldElement;   // From recent calls. Changing this causes cache flushing.
     49    std::string             mWorldElementName; // The name of above.
     50    typedef std::map<const G4VPhysicalVolume*,Relation> G4ToRelationMap_t;
     51    typedef std::map<const IDetectorElement* ,Relation> DetElemToRelationMap_t;
     52    typedef std::list<const IDetectorElement*>          ElementList_t;
     53    G4ToRelationMap_t      mG4ToRelationMap;
     54    DetElemToRelationMap_t mDetElemToRelationMap;
     55    std::vector<std::string>             mLastSearchPaths;
     56    ElementList_t                        mElementList;
     57 
     58    StatusCode GetRelation(const G4VPhysicalVolume* inVol, Relation* &outRelation );
     59    StatusCode GetRelation(const IDetectorElement* inElem, Relation* &outRelation );
     60 
     61   /// Returns -1 if incompatible, returns number that increases the better the container describes the place.
     62   int        Compatability(const ILVolume::ReplicaPath& inPlace, const ILVolume::ReplicaPath& inContainer);
     63 
     64   template < class T  >
     65   StatusCode FindObjectsInDirectory(const std::string& dirname, std::list<const T*>& outList);
     66 
     67 };




    





