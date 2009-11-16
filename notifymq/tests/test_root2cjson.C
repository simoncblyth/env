{
   // grab an example TObject 
   gSystem->Load(Form("$ABERDEEN_HOME/DataModel/lib/libAbtDataModel.%s", gSystem->GetSoExt()));
   TFile* f = TFile::Open("$ABERDEEN_HOME/DataModel/sample/run00027.root");
   TTree* t = f->Get("T") ;
   AbtRunInfo* ri = (AbtRunInfo*)(t->GetUserInfo()->At(0)) ;
   
    // the notifymqlib includes the root2cjson function that brinhs in libcJSON too 
   gSystem->Load(Form("$ENV_HOME/notifymq/lib/libnotifymq.%s",gSystem->GetSoExt()));
   
   // the interface cannot be reduced as you might imagine as : ((TObject*)ri)->Class() != ri->Class()
   cJSON* o = root2cjson( ri->Class(), ri );

   char* out = cJSON_Print(o) ;
   cout << out << endl ;

   
   cout << "sending string ... " << endl ;
   notifymq_init();

   const char* exchange = "runinfo_abtruninfo" ;
   const char* routingkey = "runinfo_abtruninfo_add" ; 

   notifymq_sendstring( exchange , routingkey , out );
   notifymq_cleanup();


   exit(0) ;




}

