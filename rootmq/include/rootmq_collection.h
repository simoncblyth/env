#ifndef rootmq_collection_h
#define rootmq_collection_h

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdint.h>
#include <amqp.h>
#include <amqp_framing.h>
#include "rootmq.h"


int rootmq_collection_init();
int rootmq_collection_cleanup();
int rootmq_collection_keys(char* buf, size_t bufsize);

int rootmq_collection_add( rootmq_basic_msg_t * msg );

rootmq_collection_qstat_t rootmq_collection_queue_stat(const char* key );
bool_t rootmq_collection_queue_updated(const char* key );
int rootmq_collection_queue_fresh(const char* key );  // returns number of msgs that have not been accessed (with _collection_get )

int rootmq_collection_queue_length( const char* key );
rootmq_basic_msg_t* rootmq_collection_get( const char* key , int n );
void rootmq_collection_dump();
void rootmq_collection_queue_configure( const char* key , rootmq_collection_observer_t observer , void* obsargs , int msgmax  );

#ifdef __cplusplus
}
#endif

#endif 
