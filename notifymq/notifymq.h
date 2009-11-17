#ifndef notifymq_h
#define notifymq_h

#ifdef __cplusplus
extern "C"
{
#endif

int notifymq_init();
int notifymq_cleanup();
int notifymq_sendstring( char const*  exchange , char const* routingkey , char const* messagebody );
int notifymq_sendbytes(  char const*  exchange , char const* routingkey , void* msgbytes , size_t msglen );
int notifymq_exchange_declare( char const* exchange , char const* exchangetype );
int notifymq_queue_bind( char const* queue, char const* exchange , char const* bindingkey );

#ifdef __cplusplus
}
#endif
#endif 
