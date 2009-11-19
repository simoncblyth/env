
class MQ : public TObject {

  private:
        TString fExchange ;  
        TString fExchangeType ;  
        TString fQueue ;  
        TString fRoutingKey ;  

  public:
     MQ( const char* exchange = "t.exchange" , 
         const char* exchangetype = "direct" , 
         const char* queue = "t.queue" , 
         const char* routingkey = "t.key" , 
         Bool_t passive = kFALSE , 
         Bool_t durable = kFALSE , 
         Bool_t auto_delete = kTRUE , 
         Bool_t exclusive = kFALSE ); 

     virtual ~MQ();
     void Send(TObject* obj );
     ClassDef(MQ , 0) // Message Queue 
};



MQ::MQ(  const char* exchange , const char* exchangetype , const char* queue , const char* routingkey, Bool_t passive , Bool_t durable , Bool_t auto_delete , Bool_t exclusive )
{

   fExchange = exchange ;
   fExchangeType = exchangetype ;
   fQueue    = queue ;
   fRoutingKey = routingkey ;

   gSystem->Load(Form("$ENV_HOME/notifymq/lib/libnotifymq.%s",gSystem->GetSoExt()));
   
   notifymq_init();
   notifymq_exchange_declare( exchange , exchangetype , passive , durable, auto_delete  ); 
   notifymq_queue_declare(    queue                   , passive , durable, exclusive, auto_delete  ); 
   notifymq_queue_bind( queue, exchange , routingkey ); 
} 

MQ::~MQ()
{
   notifymq_cleanup();
}

void MQ::Send( TObject* obj )
{
   TMessage *tm = new TMessage(kMESS_OBJECT);
   tm->WriteObject(obj);
   char *buffer     = tm->Buffer();
   int bufferLength = tm->Length();
   cout << "MQ::Send : serialized into buffer of length " << bufferLength << endl ;
   notifymq_sendbytes( fExchange.Data() , fRoutingKey.Data() , buffer , bufferLength );
}  








void test_root2message()
{
   // grab an example TObject 
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

