
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>

#include "rootmq.h"
#include "rootmq_collection.h"
#include "rootmq_utils.h"
#include <glib.h>


// http://library.gnome.org/devel/glib/stable/glib-data-types.html
static GHashTable* rootmq_collection  ;  // hash table of routing keys associated with dequeues of messages
static const int rootmq_collection_max = 10 ;

// http://library.gnome.org/devel/glib/stable/glib-Threads.html#GStaticMutex
//static GStaticMutex rootmq_collection_mutex = G_STATIC_MUTEX_INIT;
G_LOCK_DEFINE( rootmq_collection );


void rootmq_collection_hash_dumper_(gpointer key, gpointer value, gpointer user_data);
typedef struct rootmq_collection_queue_t_ { 
   char kind ;
   union {
     GQueue*     queue  ;
     GHashTable* hash  ;
     void*       other  ;
   } v ;
  rootmq_collection_qstat_t        stat ;
  rootmq_collection_observer_t observer ;
  void* obsargs ;
} rootmq_collection_queue_t ; 



typedef struct rootmq_collection_keysbuf_t_ {
    size_t bufsize ;
    char* start ;
    char* cursor ;
} rootmq_collection_keysbuf_t ;


// internal funcs, without thread protection 
rootmq_collection_queue_t* rootmq_collection_getq_( const char* key );
rootmq_collection_queue_t* rootmq_collection_getq_or_create_( const char* key );



int rootmq_collection_init()
{
    // create hash table 
    if( rootmq_dbg > 0 )
        printf("rootmq_collection_init glib (%d,%d,%d) \n", GLIB_MAJOR_VERSION , GLIB_MINOR_VERSION, GLIB_MICRO_VERSION ); // 2,4,7 on cms01
    rootmq_collection  = g_hash_table_new(g_str_hash, g_str_equal);  // funcs for : hashing, key comparison 
    return EXIT_SUCCESS ;
}

int rootmq_collection_cleanup()
{
    return EXIT_SUCCESS ;
}

// _collection setters 

int rootmq_collection_add( rootmq_basic_msg_t * msg )
{
    /*  
      attain collection lock
          pick the q within the collection based on msg->key 
          add msg to q :  
               managing q size by popping tail when too big  
               calc stats for the q
               invoke the q->observer callback with q->obsargs , msg->key and stats
      release lock
    
       this is called by rootmq_basic_collect from inside the monitor thread 
       
       
       NB the msg is constructed from fully duped components in rootmq_basic_collect
          and is regarded as fully owned by the rootmq_collection ...
          which frees it once the collection grows sufficiently to need tail popping   
    */
    
    G_LOCK(rootmq_collection);
    if(rootmq_dbg > 2) printf("rootmq_collection_add %s \n" , msg->key );
    rootmq_collection_queue_t* q =  rootmq_collection_getq_or_create_( msg->key );

    if( q == NULL ){
        printf("rootmq_collection_add : failed to getq_or_create_ [%s] \n", msg->key );
    } else {
        int msgmax = q->stat.msgmax == 0 ? rootmq_collection_max : q->stat.msgmax  ;
        guint length ;
        switch (q->kind){
            case 'Q':
                length = g_queue_get_length( q->v.queue );
                if( length == msgmax ){
                    if( rootmq_dbg > 1 )
                        printf("rootmq_collection_add : reached max %d popping tail \n" , length );
                    rootmq_basic_msg_t* d = (rootmq_basic_msg_t*)g_queue_pop_tail( q->v.queue );
                    rootmq_basic_msg_free( d );
                }
                g_queue_push_head( q->v.queue , msg );
                break ;
            case 'H':
                break ;
            default:
                break ; 
        }

        if( msg ){
            q->stat.lastadd = msg->index ;
            q->stat.received += 1 ;
            q->stat.updated = 1 ;
        }   

        //   invoke the observer callback (typically : MQ::CollectionObserver) 
        //   for the q corresponding to the message key
        if( q->observer ){
            q->observer( q->obsargs , msg->key , &(q->stat) ) ;
        }
    }
 
    G_UNLOCK(rootmq_collection);
    return EXIT_SUCCESS ;
}


rootmq_collection_qstat_t rootmq_collection_queue_stat(const char* key )
{
    G_LOCK(rootmq_collection);
    rootmq_collection_qstat_t qstat ;
    rootmq_collection_queue_t* q = rootmq_collection_getq_( key );
    qstat = q->stat ; 
    G_UNLOCK(rootmq_collection);
    return qstat ;
}

bool_t rootmq_collection_queue_updated(const char* key )
{
    /*
    
         defaults to not updated if no q 
             updated 
               : set to 1 ... when a msg is added
               : set to 0 ... when msg accessed via _get
    
    */
    G_LOCK(rootmq_collection);
    bool_t updated = 0 ;
    rootmq_collection_queue_t* q = rootmq_collection_getq_( key );
    updated = q == NULL ? 0 : q->stat.updated ;  
    G_UNLOCK(rootmq_collection);
    return updated ;
}


/*
  // Unfortunately GHashTableIter is only in newer glib-2.0 : its in 0.23 , not in 0.21

int rootmq_collection_keys(char* buf, size_t bufsize )
{   
    char* p  = buf ;
    G_LOCK(rootmq_collection);
    GHashTableIter iter;
    gpointer key, value;
    g_hash_table_iter_init (&iter, rootmq_collection);
    while (g_hash_table_iter_next (&iter, &key, &value)) {
        p += snprintf( p,  bufsize - (p - buf),  "%s ", key );
    }
    G_UNLOCK(rootmq_collection);
    
    if( bufsize < p - buf ){
        printf("rootmq_collection_keys : ERROR : bufsize %d is too small ... \n", bufsize) ;
        buf[0] = 0 ;
        return 13 ;
    } 
    return 0 ;
}
*/


void rootmq_collection_key_iter_(gpointer key, gpointer value, gpointer kb_ )
{
     rootmq_collection_keysbuf_t* kb = (rootmq_collection_keysbuf_t*)kb_ ;
     kb->cursor += snprintf( kb->cursor ,  kb->bufsize - (kb->cursor - kb->start),  "%s ", key );
}

int rootmq_collection_keys( char* buf , size_t bufsize )
{
    rootmq_collection_keysbuf_t* kb = (rootmq_collection_keysbuf_t*)malloc( sizeof( rootmq_collection_keysbuf_t ) );
    kb->start = buf ;
    kb->bufsize = bufsize ;
    kb->cursor = buf ;
    
    G_LOCK(rootmq_collection);
    g_hash_table_foreach( rootmq_collection , rootmq_collection_key_iter_ , kb );
    G_UNLOCK(rootmq_collection);
}


void rootmq_collection_dump( )
{
    G_LOCK(rootmq_collection);
    g_hash_table_foreach( rootmq_collection , rootmq_collection_hash_dumper_ , NULL );
    G_UNLOCK(rootmq_collection);
}

int rootmq_collection_queue_length( const char* key )
{
    int len = -1 ;
    G_LOCK(rootmq_collection);
    rootmq_collection_queue_t* q = rootmq_collection_getq_( key );
    if( q == NULL ){
        printf("_collection_length ERROR no q for key \"%s\" \n", key );
    } else {
        switch (q->kind){
           case 'Q':
              len =  (int)g_queue_get_length( q->v.queue );
           case 'H':
              break;
           default:
              break;
        }
    }
    G_UNLOCK(rootmq_collection);
    return len ;
}


void rootmq_collection_queue_configure( const char* key , rootmq_collection_observer_t observer , void* obsargs , int msgmax )
{
    G_LOCK(rootmq_collection);
    if(rootmq_dbg > 0) printf("rootmq_collection_queue_configure %s\n",key );
    rootmq_collection_queue_t* q =  rootmq_collection_getq_or_create_( key );
    if( q == NULL ){
        printf("_collection_add_observer ERROR failed to create q for key \"%s\" \n", key );
    } else {
        q->observer = observer ;
        q->obsargs  = obsargs ;
        q->stat.msgmax = msgmax ;
    }
    G_UNLOCK(rootmq_collection);
} 


void rootmq_collection_queue_set_maxlen( const char* key ,  int msgmax )
{
    if(rootmq_dbg > 0) printf("rootmq_collection_queue_set_msgmax %s %d\n",key, msgmax );
    G_LOCK(rootmq_collection);
    rootmq_collection_queue_t* q =  rootmq_collection_getq_or_create_( key );
    if( q == NULL ){
        printf("_collection_queue_set_msgmax ERROR failed to create q for key \"%s\" \n", key );
    } else {
        q->stat.msgmax = msgmax ;
    }
    G_UNLOCK(rootmq_collection);
}

int rootmq_collection_queue_get_maxlen( const char* key )
{
    int msgmax = 0 ;  
    if(rootmq_dbg > 0) printf("rootmq_collection_queue_get_msgmax %s \n",key );
    G_LOCK(rootmq_collection);
    rootmq_collection_queue_t* q =  rootmq_collection_getq_or_create_( key );
    if( q == NULL ){
        printf("_collection_queue_set_msgmax ERROR failed to create q for key \"%s\" \n", key );
    } else {
        msgmax = q->stat.msgmax  ;
    }
    G_UNLOCK(rootmq_collection);
}



/*
rootmq_basic_msg_t* rootmq_collection_get( const char* key , int n ) 
{
    rootmq_basic_msg_t* msg = NULL ;
    G_LOCK(rootmq_collection);
    rootmq_collection_queue_t* q = rootmq_collection_getq_( key );
    if( q == NULL ){
        printf("_collection_get ERROR no q for key \"%s\" \n", key );
    } else {
        switch (q->kind){
           case 'Q':
              msg = (rootmq_basic_msg_t*)g_queue_peek_nth( q->v.queue , n  ) ;   
           case 'H':
              break;
           default:
              break;
        }
        if( msg ){
            q->stat.updated = 0 ;    
            q->stat.read += 1 ;
            q->stat.lastread = msg->index ;
            msg->accessed += 1 ;
        }
    }
    G_UNLOCK(rootmq_collection);
    return msg ; 
}
*/

rootmq_basic_msg_t* rootmq_collection_pop( const char* key , int n ) 
{
    /*
             n > 0      pop_nth with n
             n = 0      pop_head
             n = -1     pop_tail
             n = -2     pop_nth with 0              ... checking get same as pop_head
             n = -3     pop_nth with  length - 1   .... checking get same as pop_tail

    */
    rootmq_basic_msg_t* msg = NULL ;
    G_LOCK(rootmq_collection);
    rootmq_collection_queue_t* q = rootmq_collection_getq_( key );
    if( q == NULL ){
        if( rootmq_dbg > 4 )
            printf("_collection_pop ERROR no q for key \"%s\" \n", key );
    } else {       
        int len =  (int)g_queue_get_length( q->v.queue );
        switch(n)
        {
           case  0:    msg = (rootmq_basic_msg_t*)g_queue_pop_head( q->v.queue ) ;    break ;
           case -1:    msg = (rootmq_basic_msg_t*)g_queue_pop_tail( q->v.queue ) ;    break ;
           case -2:    msg = (rootmq_basic_msg_t*)g_queue_pop_nth( q->v.queue , 0 ) ; break ;
           case -3:    msg = (rootmq_basic_msg_t*)g_queue_pop_nth( q->v.queue , len - 1 ) ;  break;  
           default:    msg = (rootmq_basic_msg_t*)g_queue_pop_nth( q->v.queue , n ) ; 
        }
    }
    G_UNLOCK(rootmq_collection);
    return msg ; 
}

rootmq_basic_msg_t* rootmq_collection_get( const char* key , int n ) 
{
    /*
          _get records the access 
          _peek doesnt (by default)
    */
    return rootmq_collection_peek( key , n , 1 );
}

rootmq_basic_msg_t* rootmq_collection_peek( const char* key , int n , int record ) 
{
    /*
             n > 0      peek_nth with n
             n = 0      peek_head
             n = -1     peek_tail
             n = -2     peek_nth with  0             ... checking get same as peek_head
             n = -3     peek_nth with  length - 1   .... checking get same as peek_tail

    */
    rootmq_basic_msg_t* msg = NULL ;
    G_LOCK(rootmq_collection);
    rootmq_collection_queue_t* q = rootmq_collection_getq_( key );
    if( q == NULL ){
        if( rootmq_dbg > 4 )
            printf("_collection_peek ERROR no q for key \"%s\" \n", key );
    } else {
        int len =  (int)g_queue_get_length( q->v.queue );
        switch(n)
        {
            case  0:    msg = (rootmq_basic_msg_t*)g_queue_peek_head( q->v.queue ) ;    break ;
            case -1:    msg = (rootmq_basic_msg_t*)g_queue_peek_tail( q->v.queue ) ;    break ;
            case -2:    msg = (rootmq_basic_msg_t*)g_queue_peek_nth( q->v.queue , 0 ) ; break ;
            case -3:    msg = (rootmq_basic_msg_t*)g_queue_peek_nth( q->v.queue , len - 1 ) ; break;  
            default:    msg = (rootmq_basic_msg_t*)g_queue_peek_nth( q->v.queue , n ) ; 
        }
    }
    
    if(record && msg){
        q->stat.updated = 0 ;    
        q->stat.read += 1 ;
        q->stat.lastread = msg->index ;
        msg->accessed += 1 ;
    }
    
    G_UNLOCK(rootmq_collection);
    return msg ; 
}


int rootmq_collection_accessed( const char* key , int n ) 
{
    /*
          the record = 0  parameter to _peek means that the access is not recorded
    */
    rootmq_basic_msg_t* msg = rootmq_collection_peek( key , n , 0 );  
    if(!msg) return -1 ;
    return msg->accessed ;
}




// private funcs  ... thread protection must be done by public interface callers of these

rootmq_collection_queue_t* rootmq_collection_getq_( const char* key )
{
    return (rootmq_collection_queue_t*)g_hash_table_lookup( rootmq_collection ,  key  );  
}

rootmq_collection_queue_t* rootmq_collection_queue_alloc_()
{
    rootmq_collection_queue_t* q = (rootmq_collection_queue_t*)malloc( sizeof( rootmq_collection_queue_t ) );
   
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

    q->stat.received = 0 ;
    q->stat.read = 0 ;
    q->stat.updated = 0 ;
    q->stat.lastread= 0 ;
    q->stat.lastadd = 0 ;
    q->stat.msgmax  = 0 ;

    q->observer = NULL ;
    q->obsargs = NULL ;

    return q ;
}

void rootmq_collection_queue_free_(rootmq_collection_queue_t* q )
{
}

void rootmq_collection_queue_dump_(rootmq_collection_queue_t* q )
{
    if(!q){
        printf("_collection_queue_dump NULL q\n");
        return ;
    }
    guint length = g_queue_get_length(q->v.queue);
    printf("_collection_queue_dump : length %d received %lld read %lld updated %d lastread %lld lastadd %lld \n", length, q->stat.received, q->stat.read, q->stat.updated, q->stat.lastread, q->stat.lastadd ); 
}

rootmq_collection_queue_t* rootmq_collection_getq_or_create_( const char* key )
{
    rootmq_collection_queue_t* q = rootmq_collection_getq_( key );
    if( q == NULL ){      
       if( rootmq_dbg > 0 )
           printf("_collection_getq_or_create : creating dq for key \"%s\" \n", key ); 

       g_hash_table_insert( rootmq_collection , g_strdup( key ) , rootmq_collection_queue_alloc_() );
       q =  rootmq_collection_getq_( key );

       // rootmq_collection_add_observer( key ,  rootmq_collection_demo_observer ); causes deadlock when done here 

    
    } else {
       if( rootmq_dbg > 1 )
           printf("_collection_getq_or_create using pre-existing dq for key \"%s\" \n", key ); 
    }
    return q ;
}

void rootmq_collection_hash_dumper_(gpointer key, gpointer value, gpointer user_data)
{
   rootmq_collection_queue_t* q = (rootmq_collection_queue_t*)value ;
   guint length = g_queue_get_length(q->v.queue);
   printf("_hash_dumper key \"%s\" length %d \n", (char*)key, length );
   rootmq_collection_queue_dump_( q );

   guint n ;
   char label[50] ;
   for( n = 0 ; n < length ; n++ ){
        switch ( q->kind ){
           case 'Q':
               sprintf( label, "queue_peek_nth %d ", n );
               rootmq_basic_msg_dump(  (rootmq_basic_msg_t*)g_queue_peek_nth(q->v.queue, n ), 0 , label  ); 
               break ;
           case 'H':
               break ;
        }     
   }

   /*
   switch ( q->kind ){
      case 'Q':
          rootmq_basic_msg_dump( (rootmq_basic_msg_t*)g_queue_peek_head( q->v.queue ) , 0 , "peek_head" ); 
          rootmq_basic_msg_dump( (rootmq_basic_msg_t*)g_queue_peek_tail( q->v.queue ) , 0 , "peek_tail" ); 
          break ;
   }
   */

}


