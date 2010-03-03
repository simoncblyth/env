
// invoke by : 
//      notifymq-
//      notifymq-cd
//      make test_root2message
//
// the kFALSE prevents starting monitor thread, just establish connection to 
// potentially remote queue and send it a message
//

void test_root2message()
{
   gSystem->Load("libnotifymq");
   MQ::Create(kFALSE);  // creates gMQ global singleton
  
   gSystem->Load("libAbtDataModel");

   TFile* f = TFile::Open("$ABERDEEN_HOME/DataModel/sample/run00027.root");
   TTree* t = f->Get("T") ;

   AbtRunInfo* ri = (AbtRunInfo*)(t->GetUserInfo()->At(0)) ;
   AbtEvent* evt = 0;
   t->SetBranchAddress( "trigger", &evt );
   Int_t n = (Int_t)t->GetEntries();
   n = 10 ;

   gMQ->SendObject( ri );

   for (Int_t i=0;i<n;i++) {
       t->GetEntry(i);
       cout << evt->GetSerialNumber() << endl ;
       gMQ->SendObject( evt );
   }   

   exit(0) ;
}

