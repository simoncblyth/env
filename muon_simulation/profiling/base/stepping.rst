stepping
==========

.. contents:: :local:


G4EventManager::DoProcessing
-------------------------------

::

    (gdb) b 'G4EventManager::DoProcessing(G4Event*)' 
    Breakpoint 1 at 0xb7b9fd94: file src/G4EventManager.cc, line 100.
    (gdb) p verboseLevel = 1
    $2 = 1
    (gdb) p verboseLevel
    $3 = 1
    (gdb) c
    Continuing.
    =====================================
      G4EventManager::ProcessOneEvent()  
    =====================================
    1 primaries are passed from G4EventTransformer.
    !!!!!!! Now start processing an event !!!!!!!
    Track (trackID 1, parentID 0) is processed with stopping code 4
    Track (trackID 57, parentID 1) is processed with stopping code 2
    ...


::

    [blyth@belle7 20130820-1318]$ pprof --list=G4EventManager::DoProcessing $(which python) base.prof
    Using local file /data1/env/local/dyb/external/Python/2.7/i686-slc5-gcc41-dbg/bin/python.
    Using local file base.prof.
    Removing _init from all stack traces.
    ROUTINE ====================== G4EventManager::DoProcessing in /data1/env/local/dyb/external/build/LCG/geant4.9.2.p01/source/event/src/G4EventManager.cc
       355 705977 Total samples (flat / cumulative)
         .      .   93: G4int G4EventManager::operator!=(const G4EventManager &right) const { }
         .      .   94: */
         .      .   95: 
         .      .   96: 
         .      .   97: 
    ---
         .      .   98: void G4EventManager::DoProcessing(G4Event* anEvent)
         .      .   99: {
         .      .  100:   G4StateManager* stateManager = G4StateManager::GetStateManager();
         .      .  101:   G4ApplicationState currentState = stateManager->GetCurrentState();
         .      .  102:   if(currentState!=G4State_GeomClosed)
         .      .  103:   {
         .      .  104:     G4Exception("G4EventManager::ProcessOneEvent",
         .      .  105:                 "IllegalApplicationState",
         .      .  106:                 JustWarning,
         .      .  107:                 "Geometry is not closed : cannot process an event.");
         .      .  108:     return;
         .      .  109:   }
         .      .  110:   currentEvent = anEvent;
         .      .  111:   stateManager->SetNewState(G4State_EventProc);
         .      .  112:   if(storetRandomNumberStatusToG4Event>1)
         .      .  113:   {
         .      .  114:     std::ostringstream oss;
         .      .  115:     CLHEP::HepRandom::saveFullState(oss);
         .      .  116:     randomNumberStatusToG4Event = oss.str();
         .      .  117:     currentEvent->SetRandomNumberStatusForProcessing(randomNumberStatusToG4Event); 
         .      .  118:   }
         .      .  119: 
         .      .  120:   // Reseting Navigator has been moved to G4Eventmanager, so that resetting
         .      .  121:   // is now done for every event.
         .      .  122:   G4ThreeVector center(0,0,0);
         .      .  123:   G4Navigator* navigator =
         .      1  124:       G4TransportationManager::GetTransportationManager()->GetNavigatorForTracking();
         .      .  125:   navigator->LocateGlobalPointAndSetup(center,0,false);
         .      .  126:                                                                                       
         .      .  127:   G4Track * track;
         .      .  128:   G4TrackStatus istop;
         .      .  129: 
         .      .  130: #ifdef G4VERBOSE
         .      .  131:   if ( verboseLevel > 0 )
         .      .  132:   {
         .      .  133:     G4cout << "=====================================" << G4endl;
         .      .  134:     G4cout << "  G4EventManager::ProcessOneEvent()  " << G4endl;
         .      .  135:     G4cout << "=====================================" << G4endl;
         .      .  136:   }
         .      .  137: #endif
         .      .  138: 
         .      .  139:   trackContainer->PrepareNewEvent();
         .      .  140: 
         .      .  141: #ifdef G4_STORE_TRAJECTORY
         .      .  142:   trajectoryContainer = 0;
         .      .  143: #endif
         .      .  144: 
         .      .  145:   sdManager = G4SDManager::GetSDMpointerIfExist();
         .      .  146:   if(sdManager)
         .      .  147:   { currentEvent->SetHCofThisEvent(sdManager->PrepareNewEvent()); }
         .      .  148: 
         .      .  149:   if(userEventAction) userEventAction->BeginOfEventAction(currentEvent);
         .      .  150: 
         .      .  151: #ifdef G4VERBOSE
         .      .  152:   if ( verboseLevel > 1 )
         .      .  153:   {
         .      .  154:     G4cout << currentEvent->GetNumberOfPrimaryVertex()
         .      .  155:          << " vertices passed from G4Event." << G4endl;
         .      .  156:   }
         .      .  157: #endif
         .      .  158: 
         .      .  159:   if(!abortRequested)
         .      .  160:   { StackTracks( transformer->GimmePrimaries( currentEvent, trackIDCounter ),true ); }
         .      .  161: 
         .      .  162: #ifdef G4VERBOSE
         .      .  163:   if ( verboseLevel > 0 )
         .      .  164:   {
         .      .  165:     G4cout << trackContainer->GetNTotalTrack() << " primaries "
         .      .  166:          << "are passed from G4EventTransformer." << G4endl;
         .      .  167:     G4cout << "!!!!!!! Now start processing an event !!!!!!!" << G4endl;
         .      .  168:   }
         .      .  169: #endif
         .      .  170:   
         .      .  171:   G4VTrajectory* previousTrajectory;
         .      .  172:   while( ( track = trackContainer->PopNextTrack(&previousTrajectory) ) != 0 )
         .      .  173:   {
         .      .  174: 
         .      .  175: #ifdef G4VERBOSE
        12     12  176:     if ( verboseLevel > 1 )
         .      .  177:     {
         .      .  178:       G4cout << "Track " << track << " (trackID " << track->GetTrackID()
         .      .  179:      << ", parentID " << track->GetParentID() 
         .      .  180:      << ") is passed to G4TrackingManager." << G4endl;
         .      .  181:     }
         .      .  182: #endif
         .      .  183: 
        16     16  184:     tracking = true;
         5 705523  185:     trackManager->ProcessOneTrack( track );
       144    148  186:     istop = track->GetTrackStatus();
         .      .  187:     tracking = false;
         .      .  188: 
         .      .  189: #ifdef G4VERBOSE
        20     20  190:     if ( verboseLevel > 0 )
         .      .  191:     {
         .      .  192:       G4cout << "Track (trackID " << track->GetTrackID()
         .      .  193:      << ", parentID " << track->GetParentID()
         .      .  194:          << ") is processed with stopping code " << istop << G4endl;
         .      .  195:     }
         .      .  196: #endif
         .      .  197: 
         .      .  198:     G4VTrajectory * aTrajectory = 0;
         .      .  199: #ifdef G4_STORE_TRAJECTORY
        66     67  200:     aTrajectory = trackManager->GimmeTrajectory();
         .      .  201: 
        10     10  202:     if(previousTrajectory)
         .      .  203:     {
         .      .  204:       previousTrajectory->MergeTrajectory(aTrajectory);
         .      .  205:       delete aTrajectory;
         .      .  206:       aTrajectory = previousTrajectory;
         .      .  207:     }
         .      .  208:     if(aTrajectory&&(istop!=fStopButAlive)&&(istop!=fSuspend))
         .      .  209:     {
         .      .  210:       if(!trajectoryContainer)
         .      .  211:       { trajectoryContainer = new G4TrajectoryContainer; 
         .      .  212:         currentEvent->SetTrajectoryContainer(trajectoryContainer); }
         .      .  213:       trajectoryContainer->insert(aTrajectory);
         .      .  214:     }
         .      .  215: #endif
         .      .  216: 
        61    159  217:     G4TrackVector * secondaries = trackManager->GimmeSecondaries();
        21     21  218:     switch (istop)
    ---
         .      .  219:     {
         .      .  220:       case fStopButAlive:
         .      .  221:       case fSuspend:
         .      .  222:         trackContainer->PushOneTrack( track, aTrajectory );
         .      .  223:         StackTracks( secondaries );


                    224         break;
                    225 
                    226       case fPostponeToNextEvent:
                    227         trackContainer->PushOneTrack( track );
                    228         StackTracks( secondaries );
                    229         break;
                    230 
                    231       case fStopAndKill:
                    232         StackTracks( secondaries );
                    233         delete track;
                    234         break;
                    235 
                    236       case fAlive:
                    237         G4cout << "Illeagal TrackStatus returned from G4TrackingManager!"
                    238              << G4endl;
                    239       case fKillTrackAndSecondaries:
                    240         //if( secondaries ) secondaries->clearAndDestroy();
                    241         if( secondaries )
                    242         {
                    243           for(size_t i=0;i<secondaries->size();i++)
                    244           { delete (*secondaries)[i]; }
                    245           secondaries->clear();
                    246         }
                    247         delete track;
                    248         break;
                    249     }
                    250   }
                    251 
                    252 #ifdef G4VERBOSE
                    253   if ( verboseLevel > 0 )
                    254   {
                    255     G4cout << "NULL returned from G4StackManager." << G4endl;
                    256     G4cout << "Terminate current event processing." << G4endl;
                    257   }
                    258 #endif
                    259 
                    260   if(sdManager)
                    261   { sdManager->TerminateCurrentEvent(currentEvent->GetHCofThisEvent()); }
                    262 
                    263   if(userEventAction) userEventAction->EndOfEventAction(currentEvent);
                    264 
                    265   stateManager->SetNewState(G4State_GeomClosed);
                    266   currentEvent = 0;
                    267   abortRequested = false;
                    268 }




G4TrackingManager::ProcessOneTrack 
------------------------------------

::

    [blyth@belle7 20130820-1318]$ pprof --list=G4TrackingManager::ProcessOneTrack $(which python) base.prof
    Using local file /data1/env/local/dyb/external/Python/2.7/i686-slc5-gcc41-dbg/bin/python.
    Using local file base.prof.
    Removing _init from all stack traces.
    ROUTINE ====================== G4TrackingManager::ProcessOneTrack in /data1/env/local/dyb/external/build/LCG/geant4.9.2.p01/source/tracking/src/G4TrackingManager.cc
      1607 705402 Total samples (flat / cumulative)
         .      .   63:   delete fpSteppingManager;
         .      .   64:   if (fpUserTrackingAction) delete fpUserTrackingAction;
         .      .   65: }
         .      .   66: 
         .      .   67: ////////////////////////////////////////////////////////////////
    ---
        16     16   68: void G4TrackingManager::ProcessOneTrack(G4Track* apValueG4Track)
         .      .   69: ////////////////////////////////////////////////////////////////
         .      .   70: {
         .      .   71: 
         .      .   72:   // Receiving a G4Track from the EventManager, this funciton has the
         .      .   73:   // responsibility to trace the track till it stops.
         1      1   74:   fpTrack = apValueG4Track;
         .      .   75:   EventIsAborted = false;
         .      .   76: 
         .      .   77:   // Clear 2ndary particle vector
         .      .   78:   //  GimmeSecondaries()->clearAndDestroy();    
         .      .   79:   //  std::vector<G4Track*>::iterator itr;
         .      .   80:   size_t itr;
         .      .   81:   //  for(itr=GimmeSecondaries()->begin();itr=GimmeSecondaries()->end();itr++){ 
        79    148   82:   for(itr=0;itr<GimmeSecondaries()->size();itr++){ 
         .      .   83:      delete (*GimmeSecondaries())[itr];
         .      .   84:   }
         3    282   85:   GimmeSecondaries()->clear();  
         .      .   86:    
        50     50   87:   if(verboseLevel>0 && (G4VSteppingVerbose::GetSilent()!=1) ) TrackBanner();
         .      .   88:   
         .      .   89:   // Give SteppingManger the pointer to the track which will be tracked 
         7  15623   90:   fpSteppingManager->SetInitialStep(fpTrack);
         .      .   91: 
         .      .   92:   // Pre tracking user intervention process.
        70     70   93:   fpTrajectory = 0;
        10     10   94:   if( fpUserTrackingAction != NULL ) {
        10    223   95:      fpUserTrackingAction->PreUserTrackingAction(fpTrack);
         .      .   96:   }
         .      .   97: #ifdef G4_STORE_TRAJECTORY
         .      .   98:   // Construct a trajectory if it is requested
        29     29   99:   if(StoreTrajectory&&(!fpTrajectory)) { 
         .      .  100:     // default trajectory concrete class object
         .      .  101:     switch (StoreTrajectory) {
         .      .  102:     default:
         .      .  103:     case 1: fpTrajectory = new G4Trajectory(fpTrack); break;
         .      .  104:     case 2: fpTrajectory = new G4SmoothTrajectory(fpTrack); break;
         .      .  105:     case 3: fpTrajectory = new G4RichTrajectory(fpTrack); break;
         .      .  106:     }
         .      .  107:   }
         .      .  108: #endif
         .      .  109: 
         .      .  110:   // Give SteppingManger the maxmimum number of processes 
         1    625  111:   fpSteppingManager->GetProcessNumber();
         .      .  112: 
         .      .  113:   // Give track the pointer to the Step
        88     91  114:   fpTrack->SetStep(fpSteppingManager->GetStep());
         .      .  115: 
         .      .  116:   // Inform beginning of tracking to physics processes 
        49   5085  117:   fpTrack->GetDefinition()->GetProcessManager()->StartTracking(fpTrack);
         .      .  118: 
         .      .  119:   // Track the particle Step-by-Step while it is alive
         .      .  120:   G4StepStatus stepStatus;
         .      .  121: 
       367    381  122:   while( (fpTrack->GetTrackStatus() == fAlive) ||
         .      .  123:          (fpTrack->GetTrackStatus() == fStopButAlive) ){
         .      .  124: 
        54     79  125:     fpTrack->IncrementCurrentStepNumber();
       364 679688  126:     stepStatus = fpSteppingManager->Stepping();
         .      .  127: #ifdef G4_STORE_TRAJECTORY
       183    183  128:     if(StoreTrajectory) fpTrajectory->
         .      .  129:                         AppendStep(fpSteppingManager->GetStep()); 
         .      .  130: #endif
        29     29  131:     if(EventIsAborted) {
         .      .  132:       fpTrack->SetTrackStatus( fKillTrackAndSecondaries );
         .      .  133:     }
         .      .  134:   }
         .      .  135:   // Inform end of tracking to physics processes 
        70   2439  136:   fpTrack->GetDefinition()->GetProcessManager()->EndTracking();
         .      .  137: 
         .      .  138:   // Post tracking user intervention process.
        54     54  139:   if( fpUserTrackingAction != NULL ) {
        41    264  140:      fpUserTrackingAction->PostUserTrackingAction(fpTrack);
         .      .  141:   }
         .      .  142: 
         .      .  143:   // Destruct the trajectory if it was created
         .      .  144: #ifdef G4VERBOSE
        31     31  145:   if(StoreTrajectory&&verboseLevel>10) fpTrajectory->ShowTrajectory();
         .      .  146: #endif
         .      .  147:   if( (!StoreTrajectory)&&fpTrajectory ) {
         .      .  148:       delete fpTrajectory;
         .      .  149:       fpTrajectory = 0;
         .      .  150:   }
         1      1  151: }
    ---
         .      .  152: 
         .      .  153: void G4TrackingManager::SetTrajectory(G4VTrajectory* aTrajectory)
         .      .  154: {
         .      .  155: #ifndef G4_STORE_TRAJECTORY
         .      .  156:   G4Exception("G4TrackingManager::SetTrajectory is invoked without G4_STORE_TRAJECTORY compilor option");
    [blyth@belle7 20130820-1318]$ 







G4SteppingManager::Stepping
-----------------------------

94% of CPU samples within `G4SteppingManager::Stepping`, for *base* muon simulation::

    [blyth@belle7 20130820-1318]$  pprof --list=G4SteppingManager::Stepping $(which python) base.prof 
    Using local file /data1/env/local/dyb/external/Python/2.7/i686-slc5-gcc41-dbg/bin/python.
    Using local file base.prof.
    Removing _init from all stack traces.

    ROUTINE ====================== G4SteppingManager::Stepping in /data1/env/local/dyb/external/build/LCG/geant4.9.2.p01/source/tracking/src/G4SteppingManager.cc
      5959 678848 Total samples (flat / cumulative)
         .      .  112: #endif
         .      .  113: }
         .      .  114: 
         .      .  115: 
         .      .  116: //////////////////////////////////////////
    ---
        42     42  117: G4StepStatus G4SteppingManager::Stepping()
         .      .  118: //////////////////////////////////////////
         .      .  119: {
         .      .  120: 
         .      .  121: //--------
         .      .  122: // Prelude
         .      .  123: //--------
         .      .  124: #ifdef G4VERBOSE
         .      .  125:             // !!!!! Verbose
        10     10  126:              if(verboseLevel>0) fVerbose->NewStep();
         .      .  127:          else 
        13     13  128:              if(verboseLevel==-1) { 
         .      .  129:              G4VSteppingVerbose::SetSilent(1);
         .      .  130:          }
         .      .  131:          else
         9    246  132:              G4VSteppingVerbose::SetSilent(0);
         .      .  133: #endif 
         .      .  134: 
         .      .  135: // Store last PostStepPoint to PreStepPoint, and swap current and nex
         .      .  136: // volume information of G4Track. Reset total energy deposit in one Step. 
       169   1759  137:    fStep->CopyPostToPreStepPoint();
       265    317  138:    fStep->ResetTotalEnergyDeposit();
         .      .  139: 
         .      .  140: // Switch next touchable in track to current one
       390   9025  141:    fTrack->SetTouchableHandle(fTrack->GetNextTouchableHandle());
         .      .  142: 
         .      .  143: // Reset the secondary particles
       317    317  144:    fN2ndariesAtRestDoIt = 0;
        13     13  145:    fN2ndariesAlongStepDoIt = 0;
        28     28  146:    fN2ndariesPostStepDoIt = 0;
         .      .  147: 
         .      .  148: //JA Set the volume before it is used (in DefineStepLength() for User Limit) 
       122    810  149:    fCurrentVolume = fStep->GetPreStepPoint()->GetPhysicalVolume();
         .      .  150: 
         .      .  151: // Reset the step's auxiliary points vector pointer
        14     30  152:    fStep->SetPointerToVectorOfAuxiliaryPoints(0);
         .      .  153: 
         .      .  154: //-----------------
         .      .  155: // AtRest Processes
         .      .  156: //-----------------
         .      .  157: 
       146    157  158:    if( fTrack->GetTrackStatus() == fStopButAlive ){
         .      .  159:      if( MAXofAtRestLoops>0 ){
         .      .  160:         InvokeAtRestDoItProcs();
         .      .  161:         fStepStatus = fAtRestDoItProc;
         .      .  162:         fStep->GetPostStepPoint()->SetStepStatus( fStepStatus );
         .      .  163:        
         .      .  164: #ifdef G4VERBOSE
         .      .  165:             // !!!!! Verbose
         .      .  166:              if(verboseLevel>0) fVerbose->AtRestDoItInvoked();
         .      .  167: #endif 
         .      .  168: 
         .      .  169:      }
         .      .  170:      // Make sure the track is killed
         .      .  171:      fTrack->SetTrackStatus( fStopAndKill );
         .      .  172:    }
         .      .  173: 
         .      .  174: //---------------------------------
         .      .  175: // AlongStep and PostStep Processes
         .      .  176: //---------------------------------
         .      .  177: 
         .      .  178: 
         .      .  179:    else{
         .      .  180:      // Find minimum Step length demanded by active disc./cont. processes
        41 197978  181:      DefinePhysicalStepLength();
         .      .  182: 
         .      .  183:      // Store the Step length (geometrical length) to G4Step and G4Track
       402    437  184:      fStep->SetStepLength( PhysicalStep );
       198    251  185:      fTrack->SetStepLength( PhysicalStep );
       104    104  186:      G4double GeomStepLength = PhysicalStep;
         .      .  187: 
         .      .  188:      // Store StepStatus to PostStepPoint
        33     59  189:      fStep->GetPostStepPoint()->SetStepStatus( fStepStatus );
         .      .  190: 
         .      .  191:      // Invoke AlongStepDoIt 
       136  31657  192:      InvokeAlongStepDoItProcs();
         .      .  193: 
         .      .  194:      // Update track by taking into account all changes by AlongStepDoIt
       247   2898  195:      fStep->UpdateTrack();
         .      .  196: 
         .      .  197:      // Update safety after invocation of all AlongStepDoIts
        63     87  198:      endpointSafOrigin= fPostStepPoint->GetPosition();
         .      .  199: //     endpointSafety=  std::max( proposedSafety - GeomStepLength, 0.);
       120    167  200:      endpointSafety=  std::max( proposedSafety - GeomStepLength, kCarTolerance);
         .      .  201: 
        69    116  202:      fStep->GetPostStepPoint()->SetSafety( endpointSafety );
         .      .  203: 
         .      .  204: #ifdef G4VERBOSE
         .      .  205:                          // !!!!! Verbose
        79     79  206:            if(verboseLevel>0) fVerbose->AlongStepDoItAllDone();
         .      .  207: #endif
         .      .  208: 
         .      .  209:      // Invoke PostStepDoIt
         8 214657  210:      InvokePostStepDoItProcs();
         .      .  211: 
         .      .  212: #ifdef G4VERBOSE
         .      .  213:                  // !!!!! Verbose
       345    345  214:      if(verboseLevel>0) fVerbose->PostStepDoItAllDone();
         .      .  215: #endif
         .      .  216:    }
         .      .  217: 
         .      .  218: //-------
         .      .  219: // Finale
         .      .  220: //-------
         .      .  221: 
         .      .  222: // Update 'TrackLength' and remeber the Step length of the current Step
        69    144  223:    fTrack->AddTrackLength(fStep->GetStepLength());
        85     91  224:    fPreviousStepSize = fStep->GetStepLength();
        21     36  225:    fStep->SetTrack(fTrack);
         .      .  226: #ifdef G4VERBOSE
         .      .  227:                          // !!!!! Verbose
         .      .  228: 
       108    108  229:            if(verboseLevel>0) fVerbose->StepInfo();
         .      .  230: #endif
         .      .  231: // Send G4Step information to Hit/Dig if the volume is sensitive
       270   2794  232:    fCurrentVolume = fStep->GetPreStepPoint()->GetPhysicalVolume();
       288    296  233:    StepControlFlag =  fStep->GetControlFlag();
        29     29  234:    if( fCurrentVolume != 0 && StepControlFlag != AvoidHitInvocation) {
         .      .  235:       fSensitive = fStep->GetPreStepPoint()->
       258    278  236:                                    GetSensitiveDetector();
        26     26  237:       if( fSensitive != 0 ) {
         1   4325  238:         fSensitive->Hit(fStep);
         .      .  239:       }
         .      .  240:    }
         .      .  241: 
         .      .  242: // User intervention process.
        39     39  243:    if( fUserSteppingAction != NULL ) {
       167 206152  244:       fUserSteppingAction->UserSteppingAction(fStep);
         .      .  245:    }
         .      .  246:    G4UserSteppingAction* regionalAction
         .      .  247:     = fStep->GetPreStepPoint()->GetPhysicalVolume()->GetLogicalVolume()->GetRegion()
      1188   2901  248:       ->GetRegionalSteppingAction();
        18     18  249:    if( regionalAction ) regionalAction->UserSteppingAction(fStep);
         .      .  250: 
         .      .  251: // Stepping process finish. Return the value of the StepStatus.
         2      2  252:    return fStepStatus;
         .      .  253: 
         7      7  254: }
    ---
         .      .  255: 
         .      .  256: ///////////////////////////////////////////////////////////
         .      .  257: void G4SteppingManager::SetInitialStep(G4Track* valueTrack)
         .      .  258: ///////////////////////////////////////////////////////////
         .      .  259: {



G4SteppingManager::InvokePostStepDoItProcs
-------------------------------------------


::

    [blyth@belle7 20130820-1318]$  pprof --list=G4SteppingManager::InvokePostStepDoItProcs $(which python) base.prof 
    Using local file /data1/env/local/dyb/external/Python/2.7/i686-slc5-gcc41-dbg/bin/python.
    Using local file base.prof.
    Removing _init from all stack traces.
    ROUTINE ====================== G4SteppingManager::InvokePostStepDoItProcs in /data1/env/local/dyb/external/build/LCG/geant4.9.2.p01/source/tracking/src/G4SteppingManager2.cc
      2027 214554 Total samples (flat / cumulative)
         .      .  469:    }
         .      .  470: 
         .      .  471: }
         .      .  472: 
         .      .  473: ////////////////////////////////////////////////////////
    ---
        19     19  474: void G4SteppingManager::InvokePostStepDoItProcs()
         .      .  475: ////////////////////////////////////////////////////////
         .      .  476: {
         .      .  477: 
         .      .  478: // Invoke the specified discrete processes
       151    151  479:    for(size_t np=0; np < MAXofPostStepLoops; np++){
         .      .  480:    //
         .      .  481:    // Note: DoItVector has inverse order against GetPhysIntVector
         .      .  482:    //       and SelectedPostStepDoItVector.
         .      .  483:    //
       606   3267  484:      G4int Cond = (*fSelectedPostStepDoItVector)[MAXofPostStepLoops-np-1];
        80     80  485:      if(Cond != InActivated){
       206    206  486:        if( ((Cond == NotForced) && (fStepStatus == fPostStepDoItProc)) ||
         .      .  487:        ((Cond == Forced) && (fStepStatus != fExclusivelyForcedProc)) ||
         .      .  488:        ((Cond == Conditionally) && (fStepStatus == fAlongStepDoItProc)) ||
         .      .  489:        ((Cond == ExclusivelyForced) && (fStepStatus == fExclusivelyForcedProc)) || 
         .      .  490:        ((Cond == StronglyForced) ) 
         .      .  491:       ) {
         .      .  492: 
        97 195545  493:      InvokePSDIP(np);
         .      .  494:        }
         .      .  495:      } //if(*fSelectedPostStepDoItVector(np)........
         .      .  496: 
         .      .  497:      // Exit from PostStepLoop if the track has been killed,
         .      .  498:      // but extra treatment for processes with Strongly Forced flag
       743    832  499:      if(fTrack->GetTrackStatus() == fStopAndKill) {
        54     54  500:        for(size_t np1=np+1; np1 < MAXofPostStepLoops; np1++){ 
        37    191  501:      G4int Cond2 = (*fSelectedPostStepDoItVector)[MAXofPostStepLoops-np1-1];
         4      4  502:      if (Cond2 == StronglyForced) {
         4  14179  503:        InvokePSDIP(np1);
         .      .  504:          }
         .      .  505:        }
         5      5  506:        break;
         .      .  507:      }
         .      .  508:    } //for(size_t np=0; np < MAXofPostStepLoops; np++){
        21     21  509: }
    ---
         .      .  510: 
         .      .  511: 
         .      .  512: 
         .      .  513: void G4SteppingManager::InvokePSDIP(size_t np)
         .      .  514: {
    [blyth@belle7 20130820-1318]$ 



G4SteppingManager::InvokePSDIP
---------------------------------

::

    [blyth@belle7 20130820-1318]$  pprof --list=G4SteppingManager::InvokePSDIP $(which python) base.prof 
    Using local file /data1/env/local/dyb/external/Python/2.7/i686-slc5-gcc41-dbg/bin/python.
    Using local file base.prof.
    Removing _init from all stack traces.
    ROUTINE ====================== G4SteppingManager::InvokePSDIP in /data1/env/local/dyb/external/build/LCG/geant4.9.2.p01/source/tracking/src/G4SteppingManager2.cc
      4888 209391 Total samples (flat / cumulative)
         .      .  508:    } //for(size_t np=0; np < MAXofPostStepLoops; np++){
         .      .  509: }
         .      .  510: 
         .      .  511: 
         .      .  512: 
    ---
       105    105  513: void G4SteppingManager::InvokePSDIP(size_t np)
         .      .  514: {
       408   2005  515:          fCurrentProcess = (*fPostStepDoItVector)[np];
         .      .  516:          fParticleChange 
       917 168266  517:             = fCurrentProcess->PostStepDoIt( *fTrack, *fStep);
         .      .  518: 
         .      .  519:          // Update PostStepPoint of Step according to ParticleChange
       238  14920  520:      fParticleChange->UpdateStepForPostStep(fStep);
         .      .  521: #ifdef G4VERBOSE
         .      .  522:                  // !!!!! Verbose
       549    549  523:            if(verboseLevel>0) fVerbose->PostStepDoItOneByOne();
         .      .  524: #endif
         .      .  525:          // Update G4Track according to ParticleChange after each PostStepDoIt
        31  17005  526:          fStep->UpdateTrack();
         .      .  527: 
         .      .  528:          // Update safety after each invocation of PostStepDoIts
       890   4360  529:          fStep->GetPostStepPoint()->SetSafety( CalculateSafety() );
         .      .  530: 
         .      .  531:          // Now Store the secondaries from ParticleChange to SecondaryList
         .      .  532:          G4Track* tempSecondaryTrack;
         .      .  533:          G4int    num2ndaries;
         .      .  534: 
       448    499  535:          num2ndaries = fParticleChange->GetNumberOfSecondaries();
         .      .  536: 
       231    231  537:          for(G4int DSecLoop=0 ; DSecLoop< num2ndaries; DSecLoop++){
        28     70  538:             tempSecondaryTrack = fParticleChange->GetSecondary(DSecLoop);
         .      .  539:    
        15     37  540:             if(tempSecondaryTrack->GetDefinition()->GetApplyCutsFlag())
         .      .  541:             { ApplyProductionCut(tempSecondaryTrack); }
         .      .  542: 
         .      .  543:             // Set parentID 
        22     24  544:             tempSecondaryTrack->SetParentID( fTrack->GetTrackID() );
         .      .  545:         
         .      .  546:         // Set the process pointer which created this track 
        10     14  547:         tempSecondaryTrack->SetCreatorProcess( fCurrentProcess );
         .      .  548: 
         .      .  549:             // If this 2ndry particle has 'zero' kinetic energy, make sure
         .      .  550:             // it invokes a rest process at the beginning of the tracking
        40     62  551:         if(tempSecondaryTrack->GetKineticEnergy() <= DBL_MIN){
         .      .  552:           G4ProcessManager* pm = tempSecondaryTrack->GetDefinition()->GetProcessManager();
         .      .  553:           if (pm->GetAtRestProcessVector()->entries()>0){
         .      .  554:             tempSecondaryTrack->SetTrackStatus( fStopButAlive );
         .      .  555:             fSecondary->push_back( tempSecondaryTrack );
         .      .  556:                 fN2ndariesPostStepDoIt++;
         .      .  557:           } else {
         .      .  558:             delete tempSecondaryTrack;
         .      .  559:           }
         .      .  560:         } else {
         3     90  561:           fSecondary->push_back( tempSecondaryTrack );
        18     18  562:               fN2ndariesPostStepDoIt++;
         .      .  563:         }
         .      .  564:          } //end of loop on secondary 
         .      .  565: 
         .      .  566:          // Set the track status according to what the process defined
       310    366  567:          fTrack->SetTrackStatus( fParticleChange->GetTrackStatus() );
         .      .  568: 
         .      .  569:          // clear ParticleChange
       221    366  570:          fParticleChange->Clear();
       404    404  571: }
    ---
         .      .  572: 
         .      .  573: #include "G4EnergyLossTables.hh"
         .      .  574: #include "G4ProductionCuts.hh"
         .      .  575: #include "G4ProductionCutsTable.hh"
         .      .  576: 




