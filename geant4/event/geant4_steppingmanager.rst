Geant4 SteppingManager
========================

`source/tracking/include/G4SteppingManager.hh`::

    092 class G4SteppingManager
    093 ///////////////////////
    094 {
    ...
    114    const G4TrackVector* GetSecondary() const;
    115    void SetUserAction(G4UserSteppingAction* apAction);
    116    G4Track* GetTrack() const;
    117    void SetVerboseLevel(G4int vLevel);
    118    void SetVerbose(G4VSteppingVerbose*);
    119    G4Step* GetStep() const;
    120    void SetNavigator(G4Navigator* value);
    ///
    ///    POSSIBLE OVERRIDE POINT ? REPLACING THE NAVIGATOR ?
    ///
    ...
    123 // Other member functions
    124 
    125    G4StepStatus Stepping();
    126       // Steers to move the give particle from the TrackingManger 
    127       // by one Step.
    128 
    129   void SetInitialStep(G4Track* valueTrack);
    130      // Sets up initial track information (enegry, position, etc) to 
    131      // the PreStepPoint of the G4Step. This funciton has to be called 
    132      // just once before the stepping loop in the "TrackingManager".
    ...
    259 
    260    G4Navigator *fNavigator;
    261 
    ...
    419   inline G4Navigator* G4SteppingManager::GetfNavigator(){
    420    return fNavigator;
    421   }
    ...
    463   inline void G4SteppingManager::SetNavigator(G4Navigator* value){
    464     fNavigator = value;
    465   }
    466 
    467   inline void G4SteppingManager::SetUserAction(G4UserSteppingAction* apAction){
    468     fUserSteppingAction = apAction;
    469   }
    470   inline G4UserSteppingAction* G4SteppingManager::GetUserAction(){
    471     return fUserSteppingAction;
    472   }



`source/tracking/src/G4SteppingManager.cc`::

    55 //////////////////////////////////////
     56 G4SteppingManager::G4SteppingManager()
     57 //////////////////////////////////////
     58   : fUserSteppingAction(0), verboseLevel(0)
     59 {
     60 
     61 // Construct simple 'has-a' related objects
     62    fStep = new G4Step();
     63    fSecondary = fStep->NewSecondaryVector();
     64    fPreStepPoint  = fStep->GetPreStepPoint();
     65    fPostStepPoint = fStep->GetPostStepPoint();
     66 #ifdef G4VERBOSE
     67    if(G4VSteppingVerbose::GetInstance()==0) {
     68      fVerbose =  new G4SteppingVerbose();
     69      G4VSteppingVerbose::SetInstance(fVerbose);
     70      fVerbose -> SetManager(this);
     71      KillVerbose = true;
     72    }
     73    else {
     74       fVerbose = G4VSteppingVerbose::GetInstance();
     75       fVerbose -> SetManager(this);
     76       KillVerbose = false;
     77    }
     78 #endif
     79    SetNavigator(G4TransportationManager::GetTransportationManager()
     80            ->GetNavigatorForTracking());
     81 
     82    fSelectedAtRestDoItVector
     83       = new G4SelectedAtRestDoItVector(SizeOfSelectedDoItVector,0);
     84    fSelectedAlongStepDoItVector
     85       = new G4SelectedAlongStepDoItVector(SizeOfSelectedDoItVector,0);
     86    fSelectedPostStepDoItVector
     87       = new G4SelectedPostStepDoItVector(SizeOfSelectedDoItVector,0);
     88 
     89    SetNavigator(G4TransportationManager::GetTransportationManager()
     90      ->GetNavigatorForTracking());
     ///  
     ///     TWICE FOR GOOD MEASURE ?
     ///
     91 





G4SteppingManager::Stepping
----------------------------


::

    115 //////////////////////////////////////////
    116 G4StepStatus G4SteppingManager::Stepping()
    117 //////////////////////////////////////////
    118 {
    ...
    134 // Store last PostStepPoint to PreStepPoint, and swap current and nex
    135 // volume information of G4Track. Reset total energy deposit in one Step. 
    136    fStep->CopyPostToPreStepPoint();
    137    fStep->ResetTotalEnergyDeposit();
    138 
    139 // Switch next touchable in track to current one
    140    fTrack->SetTouchableHandle(fTrack->GetNextTouchableHandle());
    141 
    142 // Reset the secondary particles
    143    fN2ndariesAtRestDoIt = 0;
    144    fN2ndariesAlongStepDoIt = 0;
    145    fN2ndariesPostStepDoIt = 0;
    146 
    147 //JA Set the volume before it is used (in DefineStepLength() for User Limit) 
    148    fCurrentVolume = fStep->GetPreStepPoint()->GetPhysicalVolume();
    149 
    150 // Reset the step's auxiliary points vector pointer
    151    fStep->SetPointerToVectorOfAuxiliaryPoints(0);
    152 
    ...
    ...
    ...
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
    240 
    241 // User intervention process.
    242    if( fUserSteppingAction != 0 ) {
    243       fUserSteppingAction->UserSteppingAction(fStep);
    ///
    ///         USER HOOK AFTER ALL THE ACTION
    ///
    244    }
    245    G4UserSteppingAction* regionalAction
    246     = fStep->GetPreStepPoint()->GetPhysicalVolume()->GetLogicalVolume()->GetRegion()
    247       ->GetRegionalSteppingAction();
    248    if( regionalAction ) regionalAction->UserSteppingAction(fStep);
    249 
    250 // Stepping process finish. Return the value of the StepStatus.
    251    return fStepStatus;
    252 
    252 
    253 }



::

    255 ///////////////////////////////////////////////////////////
    256 void G4SteppingManager::SetInitialStep(G4Track* valueTrack)
    257 ///////////////////////////////////////////////////////////
    258 {
    259 
    260 // Set up several local variables.
    261    PreStepPointIsGeom = false;
    262    FirstStep = true;
    263    fParticleChange = 0;
    264    fPreviousStepSize = 0.;
    265    fStepStatus = fUndefined;
    266 
    267    fTrack = valueTrack;
    268    Mass = fTrack->GetDynamicParticle()->GetMass();
    269 
    ...
    296 // Set Touchable to track and a private attribute of G4SteppingManager
    297 
    298 
    299   if ( ! fTrack->GetTouchableHandle() ) {
    300      G4ThreeVector direction= fTrack->GetMomentumDirection();
    301      fNavigator->LocateGlobalPointAndSetup( fTrack->GetPosition(),
    302                                             &direction, false, false );
    303      fTouchableHandle = fNavigator->CreateTouchableHistory();
    304 
    305      fTrack->SetTouchableHandle( fTouchableHandle );
    306      fTrack->SetNextTouchableHandle( fTouchableHandle );
    307   }else{
    308      fTrack->SetNextTouchableHandle( fTouchableHandle = fTrack->GetTouchableHandle() );
    309      G4VPhysicalVolume* oldTopVolume= fTrack->GetTouchableHandle()->GetVolume();
    310      G4VPhysicalVolume* newTopVolume=
    311      fNavigator->ResetHierarchyAndLocate( fTrack->GetPosition(),
    312         fTrack->GetMomentumDirection(),
    313     *((G4TouchableHistory*)fTrack->GetTouchableHandle()()) );
    314 //     if(newTopVolume != oldTopVolume ){
    315      if(newTopVolume != oldTopVolume || oldTopVolume->GetRegularStructureId() == 1 ) {
    316         fTouchableHandle = fNavigator->CreateTouchableHistory();
    317         fTrack->SetTouchableHandle( fTouchableHandle );
    318         fTrack->SetNextTouchableHandle( fTouchableHandle );
    319      }
    320   }
    321 // Set vertex information of G4Track at here
    322    if ( fTrack->GetCurrentStepNumber() == 0 ) {
    323      fTrack->SetVertexPosition( fTrack->GetPosition() );
    324      fTrack->SetVertexMomentumDirection( fTrack->GetMomentumDirection() );
    325      fTrack->SetVertexKineticEnergy( fTrack->GetKineticEnergy() );
    326      fTrack->SetLogicalVolumeAtVertex( fTrack->GetVolume()->GetLogicalVolume() );
    327    }



