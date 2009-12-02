
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>

#include "notifymq_collection.h"
#include "notifymq_utils.h"
#include <glib.h>


// http://library.gnome.org/devel/glib/stable/glib-data-types.html
static GHashTable* notifymq_collection  ;  // hash table of routing keys associated with dequeues of messages
static const int notifymq_collection_max = 10 ;

// http://library.gnome.org/devel/glib/stable/glib-Threads.html#GStaticMutex
//static GStaticMutex notifymq_collection_mutex = G_STATIC_MUTEX_INIT;
G_LOCK_DEFINE( notifymq_collection );


void notifymq_collection_hash_dumper_(gpointer key, gpointer value, gpointer user_data);


typedef struct notifymq_collection_queue_t_ { 
   GQueue* queue  ; 
   uint64_t received ; 
   uint64_t read ; 
   bool_t   updated ;           // since the last read 
} notifymq_collection_queue_t ; 

// internal funcs, without thread protection 
notifymq_collection_queue_t* notifymq_collection_getq_( const char* key );
notifymq_collection_queue_t* notifymq_collection_getq_or_create_( const char* key );


int notifymq_collection_init()
{
    printf("_init glib (%d,%d,%d) \n", GLIB_MAJOR_VERSION , GLIB_MINOR_VERSION, GLIB_MICRO_VERSION ); // 2,4,7 on cms01
    notifymq_collection  = g_hash_table_new(g_str_hash, g_str_equal);  // funcs for : hashing, key comparison 
    return EXIT_SUCCESS ;
}

int notifymq_collection_cleanup()
{
    return EXIT_SUCCESS ;
}

// _collection setters 

int notifymq_collection_add( notifymq_basic_msg_t * msg )
{
    G_LOCK(notifymq_collection);
    notifymq_collection_queue_t* q =  notifymq_collection_getq_or_create_( msg->key );
    guint length = g_queue_get_length( q->queue );
    if(length == notifymq_collection_max ){
        printf("_collection_add reached max %d popping tail \n" , length );
        notifymq_basic_msg_t* d = (notifymq_basic_msg_t*)g_queue_pop_tail( q->queue );
        notifymq_basic_msg_free( d );
    }
    g_queue_push_head( q->queue , msg );
    G_UNLOCK(notifymq_collection);
    return EXIT_SUCCESS ;
}

void notifymq_collection_dump( )
{
    G_LOCK(notifymq_collection);
    g_hash_table_foreach( notifymq_collection , notifymq_collection_hash_dumper_ , NULL );
    G_UNLOCK(notifymq_collection);
}

int notifymq_collection_length( const char* key )
{
    G_LOCK(notifymq_collection);
    notifymq_collection_queue_t* q = notifymq_collection_getq_( key );
    if( q == NULL )
        printf("_collection_length ERROR no q for key \"%s\" \n", key );
    int len = q == NULL ? -1 : (int)g_queue_get_length( q->queue );
    G_UNLOCK(notifymq_collection);
    return len ;
}

notifymq_basic_msg_t* notifymq_collection_get( const char* key , int n ) 
{
    G_LOCK(notifymq_collection);
    notifymq_collection_queue_t* q = notifymq_collection_getq_( key );
    if( q == NULL )
       printf("_collection_get ERROR no q for key \"%s\" \n", key );
    notifymq_basic_msg_t* msg = q == NULL ? NULL :  (notifymq_basic_msg_t*)g_queue_peek_nth( q->queue , n  ) ;   
    G_UNLOCK(notifymq_collection);
    return msg ; 
}



// private funcs  ... thread protection must be done by public interface callers of these

notifymq_collection_queue_t* notifymq_collection_getq_( const char* key )
{
    return (notifymq_collection_queue_t*)g_hash_table_lookup( notifymq_collection ,  key  );  
}

notifymq_collection_queue_t* notifymq_collection_queue_alloc_()
{
    notifymq_collection_queue_t* q = (notifymq_collection_queue_t*)malloc( sizeof( notifymq_collection_queue_t ) );
    q->queue = g_queue_new();
    q->received = 0 ;
    q->read= 0 ;
    q->updated = 0 ;
    return q ;
}

void notifymq_collection_queue_free_(notifymq_collection_queue_t* q )
{
}

notifymq_collection_queue_t* notifymq_collection_getq_or_create_( const char* key )
{
    notifymq_collection_queue_t* q = notifymq_collection_getq_( key );
    if( q == NULL ){      
       printf("_collection_getq_or_create : creating dq for key \"%s\" \n", key ); 
       g_hash_table_insert( notifymq_collection , g_strdup( key ) , notifymq_collection_queue_alloc_() );
       q =  (notifymq_collection_queue_t*)g_hash_table_lookup( notifymq_collection , key );
    } else {
       printf("_collection_getq_or_create using pre-existing dq for key \"%s\" \n", key ); 
    }
    return q ;
}

void notifymq_collection_hash_dumper_(gpointer key, gpointer value, gpointer user_data)
{
   notifymq_collection_queue_t* q = (notifymq_collection_queue_t*)value ;
   guint length = g_queue_get_length(q->queue);
   printf("_hash_dumper key \"%s\" length %d \n", (char*)key, length );
   guint n ;
   char label[50] ;
   for( n = 0 ; n < length ; n++ ){
      sprintf( label, "peek_nth %d ", n );
      notifymq_basic_msg_dump(  (notifymq_basic_msg_t*)g_queue_peek_nth(q->queue, n ), 0 , label  ); 
   }
   notifymq_basic_msg_dump( (notifymq_basic_msg_t*)g_queue_peek_head( q->queue ) , 0 , "peek_head" ); 
   notifymq_basic_msg_dump( (notifymq_basic_msg_t*)g_queue_peek_tail( q->queue ) , 0 , "peek_tail" ); 
}


