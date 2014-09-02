DetSim Configuration
=====================

On this page:

* trace physlist hookup from python DetSim config level via GaussTools and GiGa 
  to underlying Geant4 

* GiGa Gauss Geant4 hookup, orchestrated from python level
* NB **GiGaPhysListModular** comes from GaussTools not GiGa

* giga phys list 
* giga geometry control


`NuWa-trunk/dybgaudi/Simulation/DetSim/python/DetSim/Default.py`::

     01 #!/usr/bin/env python
     02 
     03 # Available physics lists
     04 physics_list_basic = [
     05     "DsPhysConsGeneral",
     06     "DsPhysConsOptical",
     07     "DsPhysConsEM",
     08     ]
     09 physics_list_nuclear = [
     10     "DsPhysConsElectroNu",
     11     "DsPhysConsHadron",
     12     "DsPhysConsIon",
     13     ]
     ..
     16 class Configure:
     17     '''
     18     Do default DetSim configuration.
     19     '''
     20 
     21     # Available geometry broken up by site
     ..
     29     giga_dayabay_items = [
     30         "/dd/Structure/Sites/db-rock",
     31         "/dd/Geometry/AdDetails/AdSurfacesAll",
     32         "/dd/Geometry/AdDetails/AdSurfacesNear",
     33         "/dd/Geometry/PoolDetails/NearPoolSurfaces",
     34         "/dd/Geometry/PoolDetails/PoolSurfacesAll",
     35         ]
     .. 
     44     def __init__(self,site="far,dayabay,lingao",
     45                  physlist = physics_list_basic+physics_list_nuclear,
     46                  use_push_algs = True,
     47                  use_sim_subseq=False ):
     48         '''
     49         Configure DetSim.  
     50 
     51         "site" can be "far", "dayabay" or "lingao".  Default is all three
     52 
     53         "physlist" specifies the physics lists.  You can use the
     54         predefined lists in DetSim.configure.: physics_list_basic and
     55         physics_list_nuclear.  Default is to use both.
     56
     57         After creating this object you may want to call historian() or
     58         unobserver() to add to their configuration.
     59 
     60         '''
     61 
     62         from GaussTools.GaussToolsConf import GiGaPhysListModular
     63         import GaudiKernel.SystemOfUnits as units
     64 
     65         # Note: we must name this with "GiGa." as it is assumed later when the
     66         # properties are looked up.  Really the action of giving it to GiGa
     67         # should take care of this.  More bugs in Configurables
     68         physics_list = GiGaPhysListModular("GiGa.GiGaPhysListModular")
     69         physics_list.CutForElectron = 100*units.micrometer
     70         physics_list.CutForPositron = 100*units.micrometer
     71         physics_list.CutForGamma = 1*units.millimeter
     /////////////////////////////////////////////////////////////////////////
     72         physics_list.PhysicsConstructors = physlist
     /////////////////////////////////////////////////////////////////////////
     73         self.physics_list = physics_list
     74 
     75         from GiGa.GiGaConf import GiGa
     76         giga = GiGa()
     77         giga.PhysicsList = physics_list

     66 def configure(argv=None):
     67     try:
     68         style = argv[0]
     69     except IndexError:
     70         Configure()
     71         return
     72 
     73     if style == 'basic':
     74         Configure(physlist = physics_list_basic)
     75         return
     76 
     77     Configure()
     78     return



`NuWa-trunk/lhcb/Sim/GaussTools/src/Components/GiGaPhysListModular.h`::

     18 class GiGaPhysListModular : public GiGaPhysListBase,
     19                             public virtual G4VModularPhysicsList
     20 {



`NuWa-trunk/lhcb/Sim/GaussTools/src/Components/GiGaPhysListModular.cpp`::

     47 GiGaPhysListModular::GiGaPhysListModular
     48 ( const std::string& type   ,
     49   const std::string& name   ,
     50   const IInterface*  parent )
     51   : G4VModularPhysicsList(),
     52     GiGaPhysListBase( type , name , parent ),
     53     m_dumpCutsTable(false)
     54 {
     55   declareProperty( "PhysicsConstructors"  , m_physconstr);
     56   declareProperty( "DumpCutsTable", m_dumpCutsTable);
     57 };
     ..
     70 StatusCode GiGaPhysListModular::initialize()
     71 {
     72   StatusCode sc=GiGaPhysListBase::initialize ();
     ..
     81   for ( std::vector<std::string>::iterator constructor = m_physconstr.begin() ;
     82         m_physconstr.end() != constructor ; ++constructor )
     83   {
     84     IGiGaPhysicsConstructor* theconstr =
     85       tool<IGiGaPhysicsConstructor>( *constructor , this ) ;
     86     if( 0 == theconstr ) { return StatusCode::FAILURE ; }
     87 
     88     // NB!!! prevent the deletion of contructors by Gaudi
     89     //for( int i = 1 ; i < 1000 ; ++i ) 
     90     //{ theconstr->addRef() ; }
     91 
     92     if( 0 == theconstr -> physicsConstructor() )
     93     { return Error ( "G4PhysicsConstructor* points to NULL!" ) ; }
     94    
     //
     // promotion of a string name like "DsPhysConsOptical" into a G4VPhysicsConstructor* instance
     //
     95     m_constructors.push_back( theconstr );
     96    
     97     // register 
     98     RegisterPhysics( theconstr -> physicsConstructor() ) ;
     99    
     00     // Print name of physics constructors registered
     01     info() << "Registered " << theconstr->name() << endmsg;
     02 
     03   }
     04  
     05   return StatusCode::SUCCESS;


`geant4.10.00.p01/source/run/include/G4VModularPhysicsList.hh`::

    090 class G4VModularPhysicsList: public virtual G4VUserPhysicsList
    091 {
    ...
    101   public:  // with description
    102     // This method will be invoked in the Construct() method. 
    103     // each particle type will be instantiated
    104     virtual void ConstructParticle();
    105 
    106     // This method will be invoked in the Construct() method.
    107     // each physics process will be instantiated and
    108     // registered to the process manager of each particle type 
    109     virtual void ConstructProcess();
    110 
    111   public: // with description
    112     // Register Physics Constructor 
    113     void RegisterPhysics(G4VPhysicsConstructor* );


`geant4.10.00.p01/source/run/src/G4VModularPhysicsList.cc`::

    131 void G4VModularPhysicsList::ConstructProcess()
    132 {
    133     G4AutoLock l(&constructProcessMutex); //Protection to be removed (A.Dotti)
    134  AddTransportation();
    135 
    136  G4PhysConstVector::iterator itr;
    137  for (itr = G4MT_physicsVector->begin(); itr!= G4MT_physicsVector->end(); ++itr) {
    138     (*itr)->ConstructProcess();
    139  }
    140 }
    ...
    144 void G4VModularPhysicsList::RegisterPhysics(G4VPhysicsConstructor* fPhysics)
    145 {
    ...
    155   G4String pName = fPhysics->GetPhysicsName();
    156   G4int pType = fPhysics->GetPhysicsType();
    157   // If physics_type is equal to 0, 
    158   // following duplication check is omitted 
    159   // This is TEMPORAL treatment.
    160   if (pType == 0) {
    161     G4MT_physicsVector->push_back(fPhysics);
    ...
    170     return;
    171   }
    172   
    173   // Check if physics with the physics_type same as one of given physics 
    174   G4PhysConstVector::iterator itr;
    175   for (itr = G4MT_physicsVector->begin(); itr!= G4MT_physicsVector->end(); ++itr) {
    176     if ( pType == (*itr)->GetPhysicsType()) break;
    177   }
    ...
    197   // register 
    198   G4MT_physicsVector->push_back(fPhysics);
    199 
    200 }   

`geant4.10.00.p01/source/run/include/G4VUserPhysicsList.hh`::

    159 class G4VUserPhysicsList
    160 {
    ...
    169   public:  // with description
    170    // Each particle type will be instantiated
    171    // This method is invoked by the RunManger 
    172    virtual void ConstructParticle() = 0;
    173 
    174    // By calling the "Construct" method, 
    175    // process manager and processes are created. 
    176    void Construct();
    177 
    178    // Each physics process will be instantiated and
    179    // registered to the process manager of each particle type 
    180    // This method is invoked in Construct method 
    181    virtual void ConstructProcess() = 0;
    182 
    183   protected: // with description
    184    //  User must invoke this method in his ConstructProcess() 
    185    //  implementation in order to insures particle transportation.
    186    void AddTransportation();
    187 
    188    //Register a process to the particle type 
    189    // according to the ordering parameter table
    190    //  'true' is returned if the process is registerd successfully
    191    G4bool RegisterProcess(G4VProcess*            process,
    192               G4ParticleDefinition*  particle);


`geant4.10.00.p01/source/run/src/G4VUserPhysicsList.cc`::

    103 class G4VUPLData
    104 {
    105     //Encapsulate the fields of class G4VUserPhysicsList
    106     //that are per-thread.
    107 public:
    108     void initialize();
    109     G4ParticleTable::G4PTblDicIterator* _theParticleIterator;
    110     G4UserPhysicsListMessenger* _theMessenger;
    111     G4PhysicsListHelper* _thePLHelper;
    112     G4bool _fIsPhysicsTableBuilt;
    113     G4int _fDisplayThreshold;
    114 };
    ...
    155 #define G4MT_thePLHelper ((this->subInstanceManager.offset[this->g4vuplInstanceID])._thePLHelper)
    ...
    946 ////////////////////////////////////////////////////////
    947 void G4VUserPhysicsList::AddTransportation()
    948 {
    949   G4MT_thePLHelper->AddTransportation();
    950 }



`geant4.10.00.p01/source/run/src/G4PhysicsListHelper.cc`::

 207 void G4PhysicsListHelper::AddTransportation()
 208 {
 ...
 220   if ( nParaWorld>0 ||
 221        useCoupledTransportation ||
 222        G4ScoringManager::GetScoringManagerIfExist()) {
 ...
 230     theTransportationProcess = new G4CoupledTransportation(verboseLevelTransport);
 231   } else {
 232     theTransportationProcess = new G4Transportation(verboseLevelTransport);
 233   }
 ...
 235   // loop over all particles in G4ParticleTable
 236   aParticleIterator->reset();
 237   while( (*aParticleIterator)() ){
 238     G4ParticleDefinition* particle = aParticleIterator->value();
 239     G4ProcessManager* pmanager = particle->GetProcessManager();
 ...
 258     // add transportation with ordering = ( -1, "first", "first" )
 259     pmanager ->AddProcess(theTransportationProcess);
 260     pmanager ->SetProcessOrderingToFirst(theTransportationProcess, idxAlongStep);
 261     pmanager ->SetProcessOrderingToFirst(theTransportationProcess, idxPostStep);
 262   }
 263 }


* Huh, I though Transportation needed to be last ? 


About Process Ordering 
-----------------------

source one
~~~~~~~~~~~~

* http://www-geant4.kek.jp/g4users/tutorial03/material/4-1.pdf

Assuming n processes, the ordering of the AlongGetPhysicalInteractionLength 
of the last processes should be::

   n-2   
   n-1  Multiple Scattering    
   n    Transportation

#. Processes return a true path length
#. Multiple Scattering effectively folds up the true path into a shorter 
   geometrical path length
#. Transportation can limit the step based on the shorter geometrical path length

source two
~~~~~~~~~~~~

* https://indico.cern.ch/event/58317/session/12/contribution/6/material/slides/0.pdf


Check the physics processes attached and their ordering::

    /particle/select e-
    /particle/processes/dump


G4ProcessManager maintains three vectors of actions :

#. **AtRest** 

   * `AtRestGetPhysicalInteractionLength()`  Returns a time, not a length 
   * `AtRestDoIt()`  process with smallest time DoIt is done

#. **AlongStep** (process ordering important for this, multiple scattering before transportation) 

   * `AlongStepGetPhysicalInteractionLength()`
   * `AlongStepDoIt()`

#. **PostStep** 

   * `PostStepGetPhysicalInteractionLength()` 
   * `PostStepDoIt()`


* http://geant4.web.cern.ch/geant4/collaboration/workshops/users2002/talks/lectures/PhysicsProcessesInGeneral.pdf


ProcessManager
-----------------

`geant4.10.00.p01/source/processes/management/include/G4ProcessManager.hh`::

     83 //  Indexes for ProcessVector
     84 enum G4ProcessVectorTypeIndex
     85 {
     86     typeGPIL = 0,   // for GetPhysicalInteractionLength 
     87     typeDoIt =1     // for DoIt
     88 };
     89 enum G4ProcessVectorDoItIndex
     90 {
     91     idxAll = -1,        // for all DoIt/GPIL 
     92     idxAtRest = 0,      // for AtRestDoIt/GPIL
     93     idxAlongStep = 1,   // for AlongStepDoIt/GPIL
     94     idxPostStep =2,     // for AlongSTepDoIt/GPIL
     95     NDoit =3
     96 };
     97
     98 //  enumeration for Ordering Parameter      
     99 enum G4ProcessVectorOrdering
     00 {
     01     ordInActive = -1,   // ordering parameter to indicate InActive DoIt
     02     ordDefault = 1000,  // default ordering parameter
     03     ordLast    = 9999   // ordering parameter to indicate the last DoIt
     04 };
     ..
     86       G4int AddProcess(
     87              G4VProcess *aProcess,
     88              G4int      ordAtRestDoIt = ordInActive,
     89              G4int      ordAlongSteptDoIt = ordInActive,
     90              G4int      ordPostStepDoIt = ordInActive
     91             );
     92       //  Add a process to the process List
     93       //  return values are index to the List. Negative return value 
     94       //  indicates that the process has not be added due to some errors
     95       //  The first argument is a pointer to process.
     96       //  Following arguments are ordering parameters of the process in 
     97       //  process vectors. If value is negative, the process is
     98       //  not added to the corresponding process vector. 
     99 
     00       //  following methods are provided for simple processes  
     01       //   AtRestProcess has only AtRestDoIt
     02       //   ContinuousProcess has only AlongStepDoIt
     03       //   DiscreteProcess has only PostStepDoIt
     04       //  If the ording parameter is not specified, the process is
     05       //  added at the end of List of process vectors 
     06       //  If a process with same ordering parameter exists, 
     07       //   this new process will be added just after processes 
     08       //   with same ordering parameter  
     09       //  (except for processes assigned to LAST explicitly )
     10       //  for both DoIt and GetPhysicalInteractionLength




::

    223       G4int GetProcessOrdering(
    224                    G4VProcess *aProcess,
    225                    G4ProcessVectorDoItIndex idDoIt
    226                                );
    227 
    228       void SetProcessOrdering(
    229                    G4VProcess *aProcess,
    230                    G4ProcessVectorDoItIndex idDoIt,
    231                    G4int      ordDoIt = ordDefault
    232                                );
    233       // Set ordering parameter for DoIt specified by typeDoIt.
    234       // If a process with same ordering parameter exists, 
    235       // this new process will be added just after processes 
    236       // with same ordering parameter  
    237       // Note: Ordering parameter will bet set to non-zero 
    238       //       even if you set  ordDoIt = 0
    239 
    240      void SetProcessOrderingToFirst(
    241                    G4VProcess *aProcess,
    242                    G4ProcessVectorDoItIndex idDoIt
    243                    );
    244       // Set ordering parameter to the first of all processes 
    245       // for DoIt specified by idDoIt.
    246       //  Note: If you use this method for two processes,
    247       //        a process called later will be first.
    248 
    249       void SetProcessOrderingToSecond(
    250                    G4VProcess *aProcess,
    251                    G4ProcessVectorDoItIndex idDoIt
    252                    );
    253       // Set ordering parameter to 1 for DoIt specified by idDoIt
    254       // and the rpocess will be added just after 
    255       // the processes with ordering parameter equal to zero
    256       //  Note: If you use this method for two processes,
    257       //        a process called later will be .



Reversal between `DoIt()` methods and `GetPhysicalInteractionLength()`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* does this explain the below, as the **First** refers to the DoIt 
  methods (transportation could limit the step) 
  BUT the distances need to be calculated in 
  reverse so the multiple scattering fold up can happen ?

::

    //  add transportation with ordering = ( -1, "first", "first" )
    pmanager->AddProcess(theTransportationProcess);
    pmanager->SetProcessOrderingToFirst(theTransportationProcess, idxAlongStep)
    pmanager->SetProcessOrderingToFirst(theTransportationProcess, idxPostStep);


* http://geant4.web.cern.ch/geant4/UserDocumentation/UsersGuides/ForApplicationDeveloper/html/ch05s02.html

Each process has two groups of methods which play an important role in
tracking, GetPhysicalInteractionLength (GPIL) and DoIt. The GPIL method gives
the step length from the current space-time point to the next space-time point.
It does this by calculating the probability of interaction based on the
process's cross section information. At the end of this step the DoIt method
should be invoked. The DoIt method implements the details of the interaction,
changing the particle's energy, momentum, direction and position, and producing
secondary tracks if required. These changes are recorded as G4VParticleChange
objects(see Particle Change).





aleniaTutorial
-----------------






