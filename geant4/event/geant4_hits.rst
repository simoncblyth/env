Geant4 Hits
============


What passes HCE to Initialize and EndOfEvent ?
------------------------------------------------


`source/digits_hits/detector/src/G4SDStructure.cc`::

    213 void G4SDStructure::Initialize(G4HCofThisEvent*HCE)
    214 {
    215   size_t i;
    216   // Broadcast to subdirectories.
    217   for( i=0; i<structure.size(); i++ )
    218   {
    219     structure[i]->Initialize(HCE);
    220   }
    221   // Initialize all detectors in this directory.
    222   for( i=0; i<detector.size(); i++ )
    223   {
    224     if(detector[i]->isActive()) detector[i]->Initialize(HCE);
    225   }
    226 }


::

     43 //  This class is exclusively used by G4SDManager for handling the tree
     44 // structure of the user's sensitive detector names.
     45 //
     46 
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
     74 






G4SensitiveDetector
--------------------

::

    [blyth@belle7 src]$ svn cp ../../DetSim/src/DsPmtSensDet.cc DsChromaPmtSensDet.cc 
    A         DsChromaPmtSensDet.cc
    [blyth@belle7 src]$ svn cp ../../DetSim/src/DsPmtSensDet.h DsChromaPmtSensDet.h 
    A         DsChromaPmtSensDet.h



* http://www-geant4.kek.jp/g4users/g4tut07/docs/SensitiveDetector.pdf


`source/digits_hits/detector/include/G4VSensitiveDetector.hh`::

    .50 class G4VSensitiveDetector
     51 {
     52 

     68   public: // with description
     69       virtual void Initialize(G4HCofThisEvent*);
     70       virtual void EndOfEvent(G4HCofThisEvent*);
     71       //  These two methods are invoked at the begining and at the end of each
     72       // event. The hits collection(s) created by this sensitive detector must
     73       // be set to the G4HCofThisEvent object at one of these two methods.

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
    100       //  This protected name vector must be filled at the constructor of the user's
    101       // concrete class for registering the name(s) of hits collection(s) being
    102       // created by this particular sensitive detector.
    103 
    104   protected:
    105       G4String SensitiveDetectorName; // detector name
    106       G4String thePathName;           // directory path
    107       G4String fullPathName;          // path + detector name
    108       G4int verboseLevel;
    109       G4bool active;
    110       G4VReadOutGeometry * ROgeometry;
    111       G4VSDFilter* filter;
    112 
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



G4SDManager
-------------


`source/digits_hits/detector/include/G4SDManager.hh`::

     ..

     43 //  This is a singleton class which manages the sensitive detectors.
     44 // The user cannot access to the constructor. The pointer of the
     45 // only existing object can be got via G4SDManager::GetSDMpointer()
     46 // static method. The first invokation of this static method makes
     47 // the singleton object.
     48 //
     49 
     50 class G4SDManager
     51 {
     52   public: // with description
     53       static G4SDManager* GetSDMpointer();
     54       // Returns the pointer to the singleton object.

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



G4OpBoundaryProcess
----------------------

To get hit from G4OpBoundaryProcess DoAbsorption needs
to be called and the rm 



::

    0164 
     165 G4VParticleChange*
     166 G4OpBoundaryProcess::PostStepDoIt(const G4Track& aTrack, const G4Step& aStep)
     167 {
     ...
     540         if ( theStatus == Detection ) InvokeSD(pStep);
     541 
     542         return G4VDiscreteProcess::PostStepDoIt(aTrack, aStep);
     543 }


    1344 G4bool G4OpBoundaryProcess::InvokeSD(const G4Step* pStep)
    1345 {
    1346   G4Step aStep = *pStep;
    1347 
    1348   aStep.AddTotalEnergyDeposit(thePhotonMomentum);
    1349 
    1350   G4VSensitiveDetector* sd = aStep.GetPostStepPoint()->GetSensitiveDetector();
    1351   if (sd) return sd->Hit(&aStep);
    1352   else return false;
    1353 }


    306 inline
    307 void G4OpBoundaryProcess::DoAbsorption()
    308 {
    309               theStatus = Absorption;
    310 
    311               if ( G4BooleanRand(theEfficiency) ) {
    ///
    ///                   need  u < theEfficiency
    ///
    312 
    313                  // EnergyDeposited =/= 0 means: photon has been detected
    314                  theStatus = Detection;
    315                  aParticleChange.ProposeLocalEnergyDeposit(thePhotonMomentum);
    316               }
    317               else {
    318                  aParticleChange.ProposeLocalEnergyDeposit(0.0);
    319               }
    320 
    321               NewMomentum = OldMomentum;
    322               NewPolarization = OldPolarization;
    323 
    324 //              aParticleChange.ProposeEnergy(0.0);
    325               aParticleChange.ProposeTrackStatus(fStopAndKill);
    326 }


Huh seems like DoAbsorption never getting called as theReflectivity is defaulting to 1.0 and theTransmittance to 0.0::

     483         else if (type == dielectric_dielectric) {
     484 
     485           if ( theFinish == polishedbackpainted ||
     486                theFinish == groundbackpainted ) {
     487              DielectricDielectric();
     488           }
     489           else {
     490              G4double rand = G4UniformRand();
     491              if ( rand > theReflectivity ) {
     492                 if (rand > theReflectivity + theTransmittance) {
     493                    DoAbsorption();
     494                 } else {
     495                    theStatus = Transmission;
     496                    NewMomentum = OldMomentum;
     497                    NewPolarization = OldPolarization;
     498                 }
     499              }



