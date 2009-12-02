
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

// accessors 
GQueue* notifymq_collection_getq_or_create( const char* key );


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
    GQueue* q =  notifymq_collection_getq_or_create( msg->key );
    guint length = g_queue_get_length( q );
    if(length == notifymq_collection_max ){
        printf("_collection_add reached max %d popping tail \n" , length );
        notifymq_basic_msg_t* d = (notifymq_basic_msg_t*)g_queue_pop_tail( q );
        notifymq_basic_msg_free( d );
    }
    g_queue_push_head( q , msg );
    return EXIT_SUCCESS ;
}

// _collection getters 

GQueue* notifymq_collection_getq( const char* key )
{
    GQueue* q = (GQueue*)g_hash_table_lookup( notifymq_collection ,  key  );  
    return q ;
}

GQueue* notifymq_collection_getq_or_create( const char* key )
{
    GQueue* q = notifymq_collection_getq( key );
    if( q == NULL ){      
       printf("_collection_getq_or_create : creating dq for key \"%s\" \n", key ); 
       g_hash_table_insert( notifymq_collection , g_strdup( key ) , g_queue_new() );
       q =  (GQueue*)g_hash_table_lookup( notifymq_collection , key );
    } else {
       printf("_collection_getq_or_create using pre-existing dq for key \"%s\" \n", key ); 
    }
    return q ;
}

void notifymq_collection_hash_dumper(gpointer key, gpointer value, gpointer user_data)
{
   GQueue* q = (GQueue*)value ;
   guint length = g_queue_get_length(q );
   printf("_hash_dumper key \"%s\" length %d \n", (char*)key, length );
   guint n ;
   char label[50] ;
   for( n = 0 ; n < length ; n++ ){
      sprintf( label, "peek_nth %d ", n );
      notifymq_basic_msg_dump(  (notifymq_basic_msg_t*)g_queue_peek_nth(q, n ), 0 , label  ); 
   }
   notifymq_basic_msg_dump( (notifymq_basic_msg_t*)g_queue_peek_head( q ) , 0 , "peek_head" ); 
   notifymq_basic_msg_dump( (notifymq_basic_msg_t*)g_queue_peek_tail( q ) , 0 , "peek_tail" ); 
}

void notifymq_collection_dump( )
{
  g_hash_table_foreach( notifymq_collection , notifymq_collection_hash_dumper , NULL );

  //notifymq_basic_msg_dump(  notifymq_collection_get( "default.routingkey" , 0 ) , 0 , "last" );
  //notifymq_basic_msg_dump(  notifymq_collection_get( "default.routingkey" , 1 ) , 0 , "penultimate" );
}

int notifymq_collection_length( const char* key )
{
   GQueue* q = notifymq_collection_getq( key );
   if( q == NULL ){
      printf("_collection_get ERROR no q for key \"%s\" \n", key );
      return -1 ;
   }
   return (int)g_queue_get_length( q );
}

notifymq_basic_msg_t* notifymq_collection_get( const char* key , int n ) 
{
   GQueue* q = notifymq_collection_getq( key );
   if( q == NULL ){
      printf("_collection_get ERROR no q for key \"%s\" \n", key );
      return NULL ;
   }
   return (notifymq_basic_msg_t*)g_queue_peek_nth( q , n  );
}



