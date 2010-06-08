
// invoke by : 
//      rootmq-
//      rootmq-cd
//      make test_sendobj
//
// the kFALSE prevents starting monitor thread, just establish connection to 
// potentially remote queue and send it a message
//

void test_sendrun()
{
   gSystem->Load("librootmq");
   MQ::Create(kFALSE);  // creates gMQ global singleton
  
   gSystem->Load("libAbtDataModel");

   TFile* f = TFile::Open("$ABERDEEN_HOME/DataModel/sample/run00027.root");
   TTree* t = f->Get("T") ;

   AbtRunInfo* ri = (AbtRunInfo*)(t->GetUserInfo()->At(0)) ;
   gMQ->SendObject( ri , "abt.test.runinfo");


   exit(0) ;
}

