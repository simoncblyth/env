
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>

#include "notifymq.h"
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
   char kind ;
   union {
     GQueue*     queue  ;
     GHashTable* hash  ;
     void*       other  ;
   } v ;
   uint64_t received ; 
   uint64_t read ; 
   uint64_t lastread ; 
   uint64_t lastadd ; 
   bool_t   updated ;           // since the last read 
} notifymq_collection_queue_t ; 


// internal funcs, without thread protection 
notifymq_collection_queue_t* notifymq_collection_getq_( const char* key );
notifymq_collection_queue_t* notifymq_collection_getq_or_create_( const char* key );


int notifymq_collection_init()
{
    if( notifymq_dbg > 0 )
        printf("_collection_init glib (%d,%d,%d) \n", GLIB_MAJOR_VERSION , GLIB_MINOR_VERSION, GLIB_MICRO_VERSION ); // 2,4,7 on cms01
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
    guint length ;
    switch (q->kind){
       case 'Q':
           length = g_queue_get_length( q->v.queue );
           if(length == notifymq_collection_max ){
               if( notifymq_dbg > 1 )
                  printf("_collection_add reached max %d popping tail \n" , length );
               notifymq_basic_msg_t* d = (notifymq_basic_msg_t*)g_queue_pop_tail( q->v.queue );
               notifymq_basic_msg_free( d );
           }
           g_queue_push_head( q->v.queue , msg );
           break ;
       case 'H':
           break ;
        default:
           break ; 
    }

    q->lastadd = msg->index ;
    q->received += 1 ;
    q->updated = 1 ;
    
    G_UNLOCK(notifymq_collection);
    return EXIT_SUCCESS ;
}

bool_t notifymq_collection_queue_updated(const char* key )
{
    G_LOCK(notifymq_collection);
    bool_t updated = 0 ;
    notifymq_collection_queue_t* q = notifymq_collection_getq_( key );
    updated = q == NULL ? 0 : q->updated ;  // defaults to not updated if no q 
    G_UNLOCK(notifymq_collection);
    return updated ;
}


void notifymq_collection_dump( )
{
    G_LOCK(notifymq_collection);
    g_hash_table_foreach( notifymq_collection , notifymq_collection_hash_dumper_ , NULL );
    G_UNLOCK(notifymq_collection);
}

int notifymq_collection_queue_length( const char* key )
{
    G_LOCK(notifymq_collection);
    notifymq_collection_queue_t* q = notifymq_collection_getq_( key );
    if( q == NULL )
        printf("_collection_length ERROR no q for key \"%s\" \n", key );

    int len ;
    switch (q->kind){
       case 'Q':
          len = q == NULL ? -1 : (int)g_queue_get_length( q->v.queue );
       case 'H':
          break;
       default:
          break;
    }

    G_UNLOCK(notifymq_collection);
    return len ;
}

notifymq_basic_msg_t* notifymq_collection_get( const char* key , int n ) 
{
    G_LOCK(notifymq_collection);
    notifymq_collection_queue_t* q = notifymq_collection_getq_( key );
    if( q == NULL )
       printf("_collection_get ERROR no q for key \"%s\" \n", key );

    notifymq_basic_msg_t* msg = NULL ;

    switch (q->kind){
       case 'Q':
          msg = q == NULL ? NULL : (notifymq_basic_msg_t*)g_queue_peek_nth( q->v.queue , n  ) ;   
       case 'H':
          break;
       default:
          break;
    }

    if( msg ){
        q->updated = 0 ;    
        q->read += 1 ;
        q->lastread = msg->index ;
    }

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
   
    char kind = *"Q" ;
    q->kind  = kind ; 
   
    switch ( q->kind ){
        case 'Q':
           q->v.queue = g_queue_new();
           break ;
        case 'H':
           q->v.hash  = g_hash_table_new( NULL , NULL );
           break ;
         default:
           q->v.other = NULL ;
       } 

    q->received = 0 ;
    q->read = 0 ;
    q->updated = 0 ;
    q->lastread= 0 ;
    q->lastadd = 0 ;

    return q ;
}

void notifymq_collection_queue_free_(notifymq_collection_queue_t* q )
{
}

void notifymq_collection_queue_dump_(notifymq_collection_queue_t* q )
{
    if(!q){
        printf("_collection_queue_dump NULL q\n");
        return ;
    }
    guint length = g_queue_get_length(q->v.queue);
    printf("_collection_queue_dump : length %d received %lld read %lld updated %d lastread %lld lastadd %lld \n", length, q->received, q->read, q->updated, q->lastread, q->lastadd ); 
}

notifymq_collection_queue_t* notifymq_collection_getq_or_create_( const char* key )
{
    notifymq_collection_queue_t* q = notifymq_collection_getq_( key );
    if( q == NULL ){      
       if( notifymq_dbg > 0 )
           printf("_collection_getq_or_create : creating dq for key \"%s\" \n", key ); 

       g_hash_table_insert( notifymq_collection , g_strdup( key ) , notifymq_collection_queue_alloc_() );
       q =  notifymq_collection_getq_( key );
    
    } else {
       if( notifymq_dbg > 1 )
           printf("_collection_getq_or_create using pre-existing dq for key \"%s\" \n", key ); 
    }
    return q ;
}

void notifymq_collection_hash_dumper_(gpointer key, gpointer value, gpointer user_data)
{
   notifymq_collection_queue_t* q = (notifymq_collection_queue_t*)value ;
   guint length = g_queue_get_length(q->v.queue);
   printf("_hash_dumper key \"%s\" length %d \n", (char*)key, length );
   notifymq_collection_queue_dump_( q );

   guint n ;
   char label[50] ;
   for( n = 0 ; n < length ; n++ ){
        switch ( q->kind ){
           case 'Q':
               sprintf( label, "queue_peek_nth %d ", n );
               notifymq_basic_msg_dump(  (notifymq_basic_msg_t*)g_queue_peek_nth(q->v.queue, n ), 0 , label  ); 
               break ;
           case 'H':
               break ;
        }     
   }

   switch ( q->kind ){
      case 'Q':
          notifymq_basic_msg_dump( (notifymq_basic_msg_t*)g_queue_peek_head( q->v.queue ) , 0 , "peek_head" ); 
          notifymq_basic_msg_dump( (notifymq_basic_msg_t*)g_queue_peek_tail( q->v.queue ) , 0 , "peek_tail" ); 
          break ;
   }


}


