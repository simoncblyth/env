
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
   //n = 5 ;

   stringstream ss ;

   Int_t pass = 0 ;
   while(kTRUE){
       pass++ ;
       cout << "test_sendobj.C : pass " << pass << " sending  " << n << " sample event objects to the queue " << endl ;
       
       gMQ->SendObject( ri , "abt.test.runinfo");
       for (Int_t i=0;i<n;i++) {
           t->GetEntry(i);
           //evt->Print("");
           gSystem->Sleep(1000);
           gMQ->SendObject( evt , "abt.test.event");
	       ss.str(""); 
           ss << "test_sendobj.C at index " << i <<  " pass "  << pass  ;
           if( i%10 == 0) gMQ->SendString( ss.str().c_str() , "abt.test.string" );
       }
       gSystem->Sleep(20);
   }   

   exit(0) ;
}

