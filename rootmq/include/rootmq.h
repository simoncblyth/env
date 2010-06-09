#ifndef rootmq_h
#define rootmq_h


#ifdef __cplusplus
extern "C"
{
#endif

#include <stdint.h>
#include <amqp.h>
#include <amqp_framing.h>

// prior to incorporation of the amqp headers  ... to be removed
typedef struct rootmq_bytes_t_ { size_t len; void *bytes; } rootmq_bytes_t;
typedef struct rootmq_props_t_ { 
   rootmq_bytes_t content_type ; 
   rootmq_bytes_t content_encoding ; 
} rootmq_props_t ; 

// to be removed .. do the threading internally 
typedef int (*receiver_t)(void* arg,  const void *msgbytes , size_t msglen , rootmq_props_t props );




// cannot expose on cint/pyROOT command line as the included types are not wrapped, but can use in compiled C++
typedef struct rootmq_basic_msg_t_ {
   uint64_t index ;
   char* key ;           // convenience copy of the routing_key  
   uint64_t accessed ;   // count the number of times the msg has been accessed eg with MQ::Get , starting at zero
   amqp_basic_deliver_t deliver ;
   amqp_basic_properties_t properties ;
   amqp_bytes_t body ;		
} rootmq_basic_msg_t ;
	
typedef int bool_t;


typedef struct rootmq_collection_qstat_t_ { 
   uint64_t msgmax  ; 
   uint64_t received ; 
   uint64_t read ; 
   uint64_t lastread ; 
   uint64_t lastadd ; 
   bool_t   updated ;           // since the last read 
} rootmq_collection_qstat_t ;

typedef int (*rootmq_collection_observer_t)(void* me, const char* key , rootmq_collection_qstat_t* args );

extern int rootmq_dbg ; 

extern int rootmq_init();
extern int rootmq_terminate();
extern int rootmq_cleanup();
extern int rootmq_sendstring( char const*  exchange , char const* routingkey , char const* messagebody );
extern int rootmq_sendbytes(  char const*  exchange , char const* routingkey , void* msgbytes , size_t msglen );
extern int rootmq_exchange_declare( char const* exchange , char const* exchangetype , bool_t passive , bool_t durable , bool_t auto_delete );
extern int rootmq_queue_bind( char const* queue, char const* exchange , char const* bindingkey );
extern int rootmq_queue_declare( char const* queue, bool_t passive , bool_t durable , bool_t exclusive , bool_t auto_delete );
//extern int rootmq_basic_consume( char const* queue , receiver_t handlebytes , void* arg  ) ;
extern int rootmq_basic_consume( char const* queue ) ;

// spin off the private monitor thread function 
extern int rootmq_basic_consume_async( char const* queue ) ;

extern const char* rootmq_get_content_type(     rootmq_basic_msg_t* msg );
extern const char* rootmq_get_content_encoding( rootmq_basic_msg_t* msg );


#ifdef __cplusplus
}
#endif
#endif 
