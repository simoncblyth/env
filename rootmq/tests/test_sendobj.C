
// invoke by : 
//      rootmq-
//      rootmq-cd
//      make test_sendobj
//
// the kFALSE prevents starting monitor thread, just establish connection to 
// potentially remote queue and send it a message
//

void test_sendobj()
{
   gSystem->Load("librootmq");
   MQ::Create(kFALSE);  // creates gMQ global singleton
  
   gSystem->Load("libAbtDataModel");

   TFile* f = TFile::Open("$ABERDEEN_HOME/DataModel/sample/run00027.root");
   TTree* t = f->Get("T") ;

   AbtRunInfo* ri = (AbtRunInfo*)(t->GetUserInfo()->At(0)) ;
   
   AbtEvent* evt = 0;
   t->SetBranchAddress( "trigger", &evt );
   Int_t n = (Int_t)t->GetEntries();
   //n = 10 ;

   Int_t pass = 0 ;
   while(kTRUE){
       pass++ ;
       cout << "test_sendobj.C : pass " << pass << " sending  " << n << " sample event objects to the queue " << endl ;
       gMQ->SendObject( ri );
       for (Int_t i=0;i<n;i++) {
           t->GetEntry(i);
           evt->Print("");
           gSystem->Sleep(2000);
           gMQ->SendObject( evt );
       }
   }   

   exit(0) ;
}

