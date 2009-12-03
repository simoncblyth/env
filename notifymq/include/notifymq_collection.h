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

bool_t notifymq_collection_queue_updated(const char* key );
int notifymq_collection_queue_length( const char* key );
notifymq_basic_msg_t* notifymq_collection_get( const char* key , int n );
void notifymq_collection_dump();
void notifymq_collection_add_observer( const char* key , notifymq_collection_observer_t observer , void* obsargs  );


#ifdef __cplusplus
}
#endif

#endif 
