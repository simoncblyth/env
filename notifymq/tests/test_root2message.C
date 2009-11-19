

void test_root2message()
{
   gSystem->Load(Form("$ENV_HOME/notifymq/lib/libnotifymq.%s", gSystem->GetSoExt()));
   gSystem->Load(Form("$ABERDEEN_HOME/DataModel/lib/libAbtDataModel.%s", gSystem->GetSoExt()));

   TFile* f = TFile::Open("$ABERDEEN_HOME/DataModel/sample/run00027.root");
   TTree* t = f->Get("T") ;

   AbtRunInfo* ri = (AbtRunInfo*)(t->GetUserInfo()->At(0)) ;
   AbtEvent* evt = 0;
   t->SetBranchAddress( "trigger", &evt );
   Int_t n = (Int_t)t->GetEntries();
   n = 10 ;

   MQ* q = new MQ();
   q->Send( ri );

   for (Int_t i=0;i<n;i++) {
       t->GetEntry(i);
       cout << evt->GetSerialNumber() << endl ;
       q->Send( evt );
   }   

   delete q ;
   exit(0) ;
}

