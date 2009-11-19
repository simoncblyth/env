
// callbacks have to be defined and set in compiled code ... not in cint 

int handlebytes( const void *msgbytes , size_t msglen )
{
   printf("inside handlebytes\n" );
   return 0; 
}


void test_basic_consume()
{
   gSystem->Load(Form("$ENV_HOME/notifymq/lib/libnotifymq.%s",gSystem->GetSoExt()));

   const char* exchange = "t.exchange" ;
   const char* exchangetype = "direct" ;
   const char* routingkey = "t.key" ; 
   const char* queue = "t.queue" ;

   notifymq_init();

   bool passive = false ;
   bool durable = false ;
   bool exclusive = false ;
   bool auto_delete = true ;


   notifymq_exchange_declare( exchange , exchangetype , passive , durable, auto_delete  ); 
   notifymq_queue_declare(    queue                   , passive , durable, exclusive, auto_delete  ); 
   notifymq_queue_bind( queue, exchange , routingkey ); 
   notifymq_basic_consume( queue , handlebytes );
   notifymq_cleanup();
   
   exit(0) ;
}

