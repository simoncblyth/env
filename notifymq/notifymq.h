#ifndef notifymq_h
#define notifymq_h

#ifdef __cplusplus
extern "C"
{
#endif

int notifymq_init();
int notifymq_cleanup();
int notifymq_sendstring( char const*  exchange , char const* routingkey , char const* messagebody );

#ifdef __cplusplus
}
#endif
#endif 
