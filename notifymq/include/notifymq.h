#ifndef notifymq_h
#define notifymq_h


#ifdef __cplusplus
extern "C"
{
#endif

#include <stdint.h>
#include <amqp.h>
#include <amqp_framing.h>

// prior to incorporation of the amqp headers  ... to be removed
typedef struct notifymq_bytes_t_ { size_t len; void *bytes; } notifymq_bytes_t;
typedef struct notifymq_props_t_ { 
   notifymq_bytes_t content_type ; 
   notifymq_bytes_t content_encoding ; 
} notifymq_props_t ; 

// to be removed .. do the threading internally 
typedef int (*receiver_t)(void* arg,  const void *msgbytes , size_t msglen , notifymq_props_t props );


// cannot expose on cint/pyROOT command line as the included types are not wrapped, but can use in compiled C++
typedef struct notifymq_basic_msg_t_ {
   uint64_t index ;
   char* key ;      // convenience copy of the routing_key  
   amqp_basic_deliver_t deliver ;
   amqp_basic_properties_t properties ;
   amqp_bytes_t body ;		
} notifymq_basic_msg_t ;
	
typedef int bool_t;


extern int notifymq_init();
extern int notifymq_cleanup();
extern int notifymq_sendstring( char const*  exchange , char const* routingkey , char const* messagebody );
extern int notifymq_sendbytes(  char const*  exchange , char const* routingkey , void* msgbytes , size_t msglen );
extern int notifymq_exchange_declare( char const* exchange , char const* exchangetype , bool_t passive , bool_t durable , bool_t auto_delete );
extern int notifymq_queue_bind( char const* queue, char const* exchange , char const* bindingkey );
extern int notifymq_queue_declare( char const* queue, bool_t passive , bool_t durable , bool_t exclusive , bool_t auto_delete );
//extern int notifymq_basic_consume( char const* queue , receiver_t handlebytes , void* arg  ) ;
extern int notifymq_basic_consume( char const* queue ) ;
extern int notifymq_basic_consume_async( char const* queue ) ;

#ifdef __cplusplus
}
#endif
#endif 
