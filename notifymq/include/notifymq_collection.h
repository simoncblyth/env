#ifndef notifmq_collection_h
#define notifymq_collection_h

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdint.h>
#include <amqp.h>
#include <amqp_framing.h>
#include "notifymq.h"

int notifymq_collection_init();
int notifymq_collection_cleanup();
int notifymq_collection_add( notifymq_basic_msg_t * msg );
void notifymq_collection_dump();
int notifymq_collection_length( const char* key );
notifymq_basic_msg_t* notifymq_collection_get( const char* key , int n );

#ifdef __cplusplus
}
#endif

#endif 
