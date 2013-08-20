stepping
==========

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








G4TrackingManager::ProcessOneTrack
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`Stepping` invoked in while loop looking at `fpTrack->GetTrackStatus()`


::

    [blyth@belle7 20130820-1318]$  pprof --list=G4TrackingManager::ProcessOneTrack $(which python) base.prof 
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






