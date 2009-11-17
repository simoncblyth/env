{
   // grab an example TObject 
   gSystem->Load(Form("$ABERDEEN_HOME/DataModel/lib/libAbtDataModel.%s", gSystem->GetSoExt()));
   TFile* f = TFile::Open("$ABERDEEN_HOME/DataModel/sample/run00027.root");
   TTree* t = f->Get("T") ;
   AbtRunInfo* ri = (AbtRunInfo*)(t->GetUserInfo()->At(0)) ;
   
    // the notifymqlib includes the root2cjson function that brinhs in libcJSON too 
   gSystem->Load(Form("$ENV_HOME/notifymq/lib/libnotifymq.%s",gSystem->GetSoExt()));
   

   TMessage *tm = new TMessage(kMESS_OBJECT);
   tm->WriteObject(ri);
   char *buffer     = tm->Buffer();
   int bufferLength = tm->Length();
   cout << "serialized into buffer of length " << bufferLength << endl ;

   const char* exchange = "root3message" ;
   const char* exchangetype = "direct" ;
   const char* routingkey = "root3message.bytes" ; 
   const char* queue = "root3message" ;

   notifymq_init();

   bool passive = false ;
   bool durable = false ;
   bool exclusive = false ;
   bool auto_delete = true ;

   notifymq_exchange_declare( exchange , exchangetype , passive , durable, auto_delete  ); 
   notifymq_queue_declare(    queue                   , passive , durable, exclusive, auto_delete  ); 
   notifymq_queue_bind( queue, exchange , routingkey ); 

   notifymq_sendbytes( exchange , routingkey , buffer , bufferLength );
   notifymq_cleanup();
   
   exit(0) ;
}

