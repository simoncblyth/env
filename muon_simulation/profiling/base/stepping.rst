stepping
==========

.. contents:: :local:


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



