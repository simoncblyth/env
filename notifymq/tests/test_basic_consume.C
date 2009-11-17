{
  
   gSystem->Load(Form("$ENV_HOME/notifymq/lib/libnotifymq.%s",gSystem->GetSoExt()));

   const char* exchange = "root5message" ;
   const char* exchangetype = "direct" ;
   const char* routingkey = "root5message.bytes" ; 
   const char* queue = "root5message" ;

   notifymq_init();

   bool passive = false ;
   bool durable = false ;
   bool exclusive = false ;
   bool auto_delete = true ;

   notifymq_exchange_declare( exchange , exchangetype , passive , durable, auto_delete  ); 
   notifymq_queue_declare(    queue                   , passive , durable, exclusive, auto_delete  ); 
   notifymq_queue_bind( queue, exchange , routingkey ); 

   notifymq_basic_consume( queue );

   notifymq_cleanup();
   
   exit(0) ;
}

