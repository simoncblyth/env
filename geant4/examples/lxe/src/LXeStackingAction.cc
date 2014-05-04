//
// ********************************************************************
// * License and Disclaimer                                           *
// *                                                                  *
// * The  Geant4 software  is  copyright of the Copyright Holders  of *
// * the Geant4 Collaboration.  It is provided  under  the terms  and *
// * conditions of the Geant4 Software License,  included in the file *
// * LICENSE and available at  http://cern.ch/geant4/license .  These *
// * include a list of copyright holders.                             *
// *                                                                  *
// * Neither the authors of this software system, nor their employing *
// * institutes,nor the agencies providing financial support for this *
// * work  make  any representation or  warranty, express or implied, *
// * regarding  this  software system or assume any liability for its *
// * use.  Please see the license in the file  LICENSE  and URL above *
// * for the full disclaimer and the limitation of liability.         *
// *                                                                  *
// * This  code  implementation is the result of  the  scientific and *
// * technical work of the GEANT4 collaboration.                      *
// * By using,  copying,  modifying or  distributing the software (or *
// * any work based  on the software)  you  agree  to acknowledge its *
// * use  in  resulting  scientific  publications,  and indicate your *
// * acceptance of all terms of the Geant4 Software license.          *
// ********************************************************************
//
#include "LXeStackingAction.hh"
#include "LXeUserEventInformation.hh"
#include "LXeSteppingAction.hh"

#include "G4ios.hh"
#include "G4ParticleDefinition.hh"
#include "G4ParticleTypes.hh"
#include "G4Track.hh"
#include "G4RunManager.hh"
#include "G4Event.hh"
#include "G4EventManager.hh"

#include "TMessage.h"

#include "ChromaPhotonList.hh"
#include "MyTMessage.hh"


#include <stdlib.h>
#include <zmq.h>
#include <assert.h>


LXeStackingAction::LXeStackingAction() : fPhotonList(NULL), fPhotonList2(NULL), fContext(NULL), fRequester(NULL) 
{
  G4cout << "LXeStackingAction::LXeStackingAction " <<  G4endl;   
  
  fPhotonList = new ChromaPhotonList ;   

  char* CONFIG = getenv("LXE_CLIENT_CONFIG") ;
  if(!CONFIG) return ; 

  G4cout << "LXeStackingAction::LXeStackingAction CONFIG " << CONFIG << G4endl;   

  fContext = zmq_ctx_new ();
  fRequester = zmq_socket (fContext, ZMQ_REQ);

  int rc = zmq_connect (fRequester, CONFIG );
  assert( rc == 0); 
}



LXeStackingAction::~LXeStackingAction()
{
  G4cout << "LXeStackingAction::~LXeStackingAction " <<  G4endl;   

  delete fPhotonList ;  

  if(fRequester != NULL){
      G4cout << "Close fRequester " <<  G4endl;   
      zmq_close (fRequester);
  }
  if(fContext != NULL){
       G4cout << "Destroy fContext " <<  G4endl;   
       zmq_ctx_destroy(fContext); 
  }

}

void LXeStackingAction::SendPhotonList()
{
   /*
   http://dayabay.phys.ntu.edu.tw/tracs/env/wiki/RootMQ
   http://dayabay.phys.ntu.edu.tw/tracs/env/browser/trunk/rootmq/src/MQ.cc
   */
   G4cout << "SendPhotonList " <<  G4endl;   

   TMessage* tmsg = new TMessage(kMESS_OBJECT);
   tmsg->WriteObject(fPhotonList);
   char *buf     = tmsg->Buffer();
   int   bufLen = tmsg->Length();  

   int rc ; 
   zmq_msg_t zmsg;

   rc = zmq_msg_init_size (&zmsg, bufLen);
   assert (rc == 0);
   memcpy(zmq_msg_data (&zmsg), buf, bufLen );   // TODO : check for zero copy approaches

   rc = zmq_msg_send (&zmsg, fRequester, 0);
   if (rc == -1) {
       int err = zmq_errno();
       printf ("Error occurred during zmq_msg_send : %s\n", zmq_strerror(err));
       abort (); 
   }

   G4cout << "SendPhotonList sent bytes: " << rc <<  G4endl;   

}

void LXeStackingAction::ReceivePhotonList()
{
    G4cout << "ReceivePhotonList waiting..." <<  G4endl;   
    zmq_msg_t msg;

    int rc = zmq_msg_init (&msg); 
    assert (rc == 0);

    rc = zmq_msg_recv (&msg, fRequester, 0);   
    assert (rc != -1);

    size_t size = zmq_msg_size(&msg); 
    void* data = zmq_msg_data(&msg) ;

    G4cout << "ReceivePhotonList received bytes: " << size <<  G4endl;   

    TObject* obj = NULL ; 

    MyTMessage* tmsg = new MyTMessage( data , size ); 
    assert( tmsg->What() == kMESS_OBJECT ); 

    TClass* kls = tmsg->GetClass();
    obj = tmsg->ReadObject(kls);

    fPhotonList2 = (ChromaPhotonList*)obj ; 


    zmq_msg_close (&msg);
}

void LXeStackingAction::ComparePhotonLists(ChromaPhotonList* a, ChromaPhotonList* b)
{
    
    a->Details();
    b->Details();
}




void LXeStackingAction::CollectPhoton(const G4Track* aPhoton )
{
   G4ParticleDefinition* pd = aPhoton->GetDefinition();
   assert( pd->GetParticleName() == "opticalphoton" );

   G4ThreeVector pos = aPhoton->GetPosition()/mm ; 
   G4ThreeVector dir = aPhoton->GetMomentumDirection() ; 
   G4ThreeVector pol = aPhoton->GetPolarization() ;
   float time = aPhoton->GetGlobalTime()/ns ;
   float wavelength = (h_Planck * c_light / aPhoton->GetKineticEnergy()) / nanometer ;

   fPhotonList->AddPhoton( pos, dir, pol, time, wavelength );

}

G4ClassificationOfNewTrack
LXeStackingAction::ClassifyNewTrack(const G4Track * aTrack){
 
  G4cout << "LXeStackingAction::ClassifyNewTrack TrackID " << aTrack->GetTrackID() << " ParentID " << aTrack->GetParentID() <<  G4endl;   

  G4bool is_op = aTrack->GetDefinition()==G4OpticalPhoton::OpticalPhotonDefinition() ; 
  G4bool is_secondary = aTrack->GetParentID()>0 ; 

  G4EventManager* evtmgr = G4EventManager::GetEventManager();
  const G4Event* event = evtmgr->GetConstCurrentEvent() ; 
  LXeUserEventInformation* eventInformation = (LXeUserEventInformation*)event->GetUserInformation();
  
  if(is_op){ 

      CollectPhoton( aTrack );
      fPhotonList->Print();

      if(is_secondary){
         G4String procname = aTrack->GetCreatorProcess()->GetProcessName() ;
         G4cout << "LXeStackingAction::ClassifyNewTrack OP Secondary from " << procname << G4endl;  

         //Count what process generated the optical photons
         if(procname=="Scintillation") eventInformation->IncPhotonCount_Scint();
         else if(procname=="Cerenkov") eventInformation->IncPhotonCount_Ceren();
      }
  }
  return fUrgent;
}

void LXeStackingAction::NewStage(){

  G4cout << "LXeStackingAction::NewStage" << G4endl;   
  SendPhotonList();
  ReceivePhotonList();
  ComparePhotonLists( fPhotonList, fPhotonList2 );

}

void LXeStackingAction::PrepareNewEvent(){ 

  G4cout << "LXeStackingAction::PrepareNewEvent" << G4endl;   

}








