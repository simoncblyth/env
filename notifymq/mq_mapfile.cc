#include <iostream>
using namespace std ;

#include "MQ.h"
#include "TObject.h"
#include "TObjString.h"
#include "TString.h"
#include "TMapFile.h"
#include "TTree.h"

#include "AbtRunInfo.h"
#include "AbtEvent.h"
#include "AbtResponse.h"

// see $ROOTSYS/tutorials/net/hprod.C  mapfiles can be opened from another process
static TMapFile* mfile = NULL ;
static TTree* tree = NULL ;
static AbtEvent* evt = NULL ;


// callbacks must be defined and set in compiled code, not from cint 
int handlebytes( const void *msgbytes , size_t msglen )
{
   cout <<  "handlebytes received msglen "  << msglen << endl ; 
   TObject* obj = MQ::Receive( msgbytes , msglen );
   if ( obj == NULL ){
       cout << "received NULL obj " << endl ;
   } else {
       TString kln = obj->ClassName();
       cout << kln.Data() ; 
       if( kln == "TObjString" ){      

          cout << ((TObjString*)obj)->GetString() << endl; 

       } else if( kln == "AbtRunInfo" ){       

          AbtRunInfo* ari = (AbtRunInfo*)obj ;
          ari->Print() ;
          tree->GetUserInfo()->Add(ari);

       } else if ( kln == "AbtEvent" ){     

          evt = (AbtEvent*)obj ;
          evt->Print(); 
          tree->Fill();
          tree->AutoSave("SaveSelf");

          mfile->Print();
          mfile->Update();       
          mfile->ls(); 

       } else {
          cout << "SKIPPING received obj of class " << kln.Data() << endl ;
       }
   }
   return 0; 
}


int main(int argc, char const * const *argv) 
{
   TMapFile::SetMapAddress(0xb46a5000);
   mfile = TMapFile::Create("mq_mapfile.map","RECREATE", 1000000, "Demo memory mapped file ");
   tree = new TTree("T","tree in a mapfile");
   evt  = new AbtEvent ;
   tree->Branch("trigger", "AbtEvent", &evt );
   tree->SetCircular( 1000 );    //  http://root.cern.ch/phpBB2/viewtopic.php?t=7964&highlight=ttree+circular  

   MQ* q = new MQ();
   q->Wait( handlebytes );
   delete q ;
   return 0;
}

  


