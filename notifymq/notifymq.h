#ifndef notifymq_h
#define notifymq_h

#ifdef __cplusplus
extern "C"
{
#endif


typedef int bool_t;

int notifymq_init();
int notifymq_cleanup();
int notifymq_sendstring( char const*  exchange , char const* routingkey , char const* messagebody );
int notifymq_sendbytes(  char const*  exchange , char const* routingkey , void* msgbytes , size_t msglen );
int notifymq_exchange_declare( char const* exchange , char const* exchangetype , bool_t passive , bool_t durable , bool_t auto_delete );
int notifymq_queue_bind( char const* queue, char const* exchange , char const* bindingkey );
int notifymq_queue_declare( char const* queue, bool_t passive , bool_t durable , bool_t exclusive , bool_t auto_delete );

#ifdef __cplusplus
}
#endif
#endif 
