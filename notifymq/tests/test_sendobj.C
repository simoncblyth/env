
// invoke by : 
//      notifymq-
//      notifymq-cd
//      make test_root2message
//
// the kFALSE prevents starting monitor thread, just establish connection to 
// potentially remote queue and send it a message
//

void test_sendobj()
{
   gSystem->Load("libnotifymq");
   MQ::Create(kFALSE);  // creates gMQ global singleton
  
   gSystem->Load("libAbtDataModel");

   TFile* f = TFile::Open("$ABERDEEN_HOME/DataModel/sample/run00027.root");
   TTree* t = f->Get("T") ;

   //AbtRunInfo* ri = (AbtRunInfo*)(t->GetUserInfo()->At(0)) ;
   //gMQ->SendObject( ri );
   
   AbtEvent* evt = 0;
   t->SetBranchAddress( "trigger", &evt );
   Int_t n = (Int_t)t->GetEntries();
   //n = 10 ;

   cout << "sending event objects to the queue " << n << endl ;

   for (Int_t i=0;i<n;i++) {
       t->GetEntry(i);
       evt->Print("");
       gSystem->Sleep(2000);
       gMQ->SendObject( evt );
   }   

   exit(0) ;
}

