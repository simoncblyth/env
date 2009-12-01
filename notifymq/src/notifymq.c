

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>

#include <stdint.h>
#include <amqp.h>
#include <amqp_framing.h>

#include <unistd.h>
#include "private.h"
#include "example_utils.h"

#include <glib.h>

#include "notifymq.h"

static int sockfd ;
static amqp_connection_state_t conn ;
extern void amqp_dump(void const *buffer, size_t len);

static GHashTable* notifymq_collection  ;  // hash table of routing keys associated with dequeues of messages
static const int notifymq_collection_max = 10 ;

static uint64_t notifymq_msg_index = 0 ; 
static int notifymq_dbg = 0 ; 

typedef struct notifymq_basic_msg_t_ {
   uint64_t index ;
   char* key ;      // convenience copy of the routing_key  
   amqp_basic_deliver_t deliver ;
   amqp_basic_properties_t properties ;
   amqp_bytes_t body ;		
} notifymq_basic_msg_t ;
	

// deprecated ... it allocates every time
char* notifymq_get_content_type( amqp_basic_properties_t* props );

// duplicators 
amqp_basic_deliver_t    notifymq_basic_deliver_dup( const amqp_basic_deliver_t src );
amqp_basic_properties_t notifymq_basic_properties_dup( const amqp_basic_properties_t src );
amqp_decimal_t          notifymq_decimal_dup( const amqp_decimal_t src );
amqp_table_t            notifymq_table_dup( const amqp_table_t src );
notifymq_basic_msg_t*   notifymq_basic_msg_dup( uint64_t index , amqp_bytes_t* body ,  amqp_basic_deliver_t* deliver , amqp_basic_properties_t* props );

// dumpers 
void notifymq_basic_deliver_dump( const amqp_basic_deliver_t* d );
void notifymq_basic_properties_dump( const amqp_basic_properties_t* p );
void notifymq_table_dump( const amqp_table_t* t );
void notifymq_basic_msg_dump( const notifymq_basic_msg_t* msg, int verbosity  );

// deallocaters
void notifymq_basic_msg_free( notifymq_basic_msg_t* msg );
void notifymq_basic_deliver_free( amqp_basic_deliver_t* src );
void notifymq_basic_properties_free( amqp_basic_properties_t* src );



char* notifymq_getstr_alloc( amqp_bytes_t b ) {
    char* buf ;
    buf = (char*)malloc( b.len );
    buf[0] = 0 ;
    if( b.bytes != NULL ){ 
       memcpy( buf , b.bytes , b.len );   // cannot use strcpy...  as .bytes from amqp_bytes_t is not null-terminated
       buf[b.len] = 0 ;                   // null termination 
    }
    return buf ;
}

int notifymq_getstr( amqp_bytes_t b , char* buf , size_t max  ) {
    if(b.len > max || b.bytes == 0) return EXIT_FAILURE ;
    memcpy( buf , b.bytes , b.len );   // cannot use strcpy...  as .bytes from amqp_bytes_t is not null-terminated
    buf[b.len] = 0 ;                   // null termination 
    return EXIT_SUCCESS ;
}


int notifymq_collection_add( notifymq_basic_msg_t * msg )
{

    gchar* k = g_strdup( msg->key );
    GQueue* q = (GQueue*)g_hash_table_lookup( notifymq_collection , k );  
    if( q == NULL ){      
       printf("_collection_add creating dq for key \"%s\" \n", msg->key ); 
       g_hash_table_insert( notifymq_collection , k , g_queue_new() );
       q =  (GQueue*)g_hash_table_lookup( notifymq_collection , k );
    } else {
       printf("_collection_add using pre-existing dq for key \"%s\" \n", msg->key ); 
    }
    guint length = g_queue_get_length( q );
    if(length == notifymq_collection_max ){
        printf("_collection_add reached max %d popping tail \n" , length );
        notifymq_basic_msg_t* d = (notifymq_basic_msg_t*)g_queue_pop_tail( q );
        notifymq_basic_msg_free( d );
    }
    g_queue_push_head( q , msg );
    return EXIT_SUCCESS ;
}


void notifymq_hash_dumper(gpointer key, gpointer value, gpointer user_data)
{
   GQueue* q = (GQueue*)value ;
   guint length = g_queue_get_length(q );
   printf("_collection_dump key \"%s\" length %d \n", (char*)key, length );

   notifymq_basic_msg_t* msg = NULL ; 
   guint n ;
   for( n = 0 ; n < length ; n++ ){
      msg = (notifymq_basic_msg_t*)g_queue_peek_nth( q , n );
      notifymq_basic_msg_dump( msg , 0 ); 
   }
}

void notifymq_collection_dump()
{
   g_hash_table_foreach( notifymq_collection , notifymq_hash_dumper , NULL );
}


int notifymq_basic_collect( amqp_bytes_t* body ,  amqp_basic_deliver_t* deliver , amqp_basic_properties_t* props )
{
    notifymq_basic_msg_t* msg = notifymq_basic_msg_dup( notifymq_msg_index , body , deliver , props );

    // test clean up
    notifymq_basic_msg_t* msg2 = notifymq_basic_msg_dup( notifymq_msg_index , body , deliver , props );
    notifymq_basic_msg_free( msg2 );

    notifymq_msg_index++ ;     // global received message index within this run 
    notifymq_basic_msg_dump( msg , 3 );

    notifymq_collection_add( msg );
    notifymq_collection_dump();

    return EXIT_SUCCESS ;
}

notifymq_basic_msg_t* notifymq_basic_msg_dup( uint64_t index , amqp_bytes_t* body ,  amqp_basic_deliver_t* deliver , amqp_basic_properties_t* props )
{
    // full duplication with new allocations, collecting msg and its metadata into single structure  
    notifymq_basic_msg_t* msg = malloc( sizeof(notifymq_basic_msg_t) );
    msg->index   = index ;
    msg->body    = amqp_bytes_malloc_dup( *body );
    msg->deliver = notifymq_basic_deliver_dup( *deliver ); 
    msg->properties = notifymq_basic_properties_dup( *props ) ;
    msg->key     = notifymq_getstr_alloc( msg->deliver.routing_key );  // convenience copy as null terminated string needed 
    return msg ;
}

void notifymq_basic_msg_dump( const notifymq_basic_msg_t* msg , int verbosity )
{
    printf("notifymq_msg_dump .index %lld .key \"%s\" verbosity %d \n", msg->index , msg->key, verbosity  );
    if(verbosity > 2)
       amqp_dump( msg->body.bytes , msg->body.len  );   
    if(verbosity > 1)
       notifymq_basic_deliver_dump( &(msg->deliver) );
    if(verbosity > 1)
       notifymq_basic_properties_dump( &(msg->properties) );  
}

void notifymq_basic_msg_free( notifymq_basic_msg_t* msg )
{
    AMQP_BYTES_FREE( msg->body );
    notifymq_basic_deliver_free( &(msg->deliver) ) ;
    notifymq_basic_properties_free( &(msg->properties) );  
    free( msg->key );
    msg = NULL ;
}


int notifymq_init()
{
    int rc = private_init();
    if(rc != EXIT_SUCCESS) return rc ;
   
    char const* hostname = private_lookup("AMQP_SERVER");
    int port = atoi(private_lookup("AMQP_PORT"));
    char const* user = private_lookup("AMQP_USER");
    char const* password = private_lookup("AMQP_PASSWORD");
    char const* vhost = private_lookup("AMQP_VHOST");
    notifymq_dbg = atoi( private_lookup_default("NOTIFYMQ_DBG", "0" )) ;

    if(notifymq_dbg > 0 ) 
        printf("notifymq_init : INFO debug level NOTIFYMQ_DBG is at level :[%d] \n", notifymq_dbg );


    rc = die_on_error(sockfd = amqp_open_socket(hostname, port), "Opening socket");
    if(rc != EXIT_SUCCESS) return rc ;

    conn = amqp_new_connection();
    amqp_set_sockfd(conn, sockfd);
    die_on_amqp_error(amqp_login(conn, vhost , 0, 131072, 0, AMQP_SASL_METHOD_PLAIN, user, password), "Logging in");
    amqp_channel_open(conn, 1);
    die_on_amqp_error(amqp_rpc_reply, "Opening channel");

    printf("_init glib (%d,%d,%d) \n", GLIB_MAJOR_VERSION , GLIB_MINOR_VERSION, GLIB_MICRO_VERSION ); // 2,4,7 on cms01
    notifymq_collection  = g_hash_table_new(g_str_hash, g_str_equal);  // funcs for : hashing, key comparison 

    return EXIT_SUCCESS ;
}


int notifymq_queue_declare( char const* queue, bool_t passive, bool_t durable, bool_t exclusive, bool_t auto_delete )
{
    amqp_queue_declare(conn, 1, 
                       amqp_cstring_bytes(queue) , 
                       passive , durable, exclusive, auto_delete , 
                       AMQP_EMPTY_TABLE );
    die_on_amqp_error(amqp_rpc_reply, "Declaring queue");
    return EXIT_SUCCESS ;
} 


int notifymq_queue_bind( char const* queue, char const* exchange , char const* bindingkey )
{
    amqp_queue_bind(conn, 1,
     		    amqp_cstring_bytes(queue),
     		    amqp_cstring_bytes(exchange),
     		    amqp_cstring_bytes(bindingkey),
     		    AMQP_EMPTY_TABLE);
    die_on_amqp_error(amqp_rpc_reply, "Binding queue");
    return EXIT_SUCCESS ;
}


int notifymq_exchange_declare( char const* exchange , char const* exchangetype , bool_t passive , bool_t durable , bool_t auto_delete )
{
    amqp_exchange_declare(conn, 1, 
                          amqp_cstring_bytes(exchange), 
                          amqp_cstring_bytes(exchangetype),
     			  passive, durable, auto_delete, 
                          AMQP_EMPTY_TABLE);
    die_on_amqp_error(amqp_rpc_reply, "Declaring exchange");
    return EXIT_SUCCESS ;
}





void notifymq_basic_deliver_dump( const amqp_basic_deliver_t* d )
{
    printf("basic_deliver_dump delivery_tag %lld redelivered %d \n", d->delivery_tag, d->redelivered );
    printf("\t.consumer_tag\t\t%.*s\n",  d->consumer_tag.len, (char*)d->consumer_tag.bytes );
    printf("\t.exchange\t\t%.*s\n",      d->exchange.len,     (char*)d->exchange.bytes );
    printf("\t.routing_key\t\t%.*s\n",   d->routing_key.len,  (char*)d->routing_key.bytes );
}

amqp_basic_deliver_t notifymq_basic_deliver_dup( const amqp_basic_deliver_t src )
{
    amqp_basic_deliver_t dest ;
    dest.consumer_tag = amqp_bytes_malloc_dup( src.consumer_tag );
    dest.delivery_tag = src.delivery_tag ;
    dest.redelivered  = src.redelivered ;
    dest.exchange     = amqp_bytes_malloc_dup( src.exchange );
    dest.routing_key  = amqp_bytes_malloc_dup( src.routing_key );
    return dest ;
}

void notifymq_basic_deliver_free( amqp_basic_deliver_t* src )
{
    AMQP_BYTES_FREE( src->consumer_tag );
    AMQP_BYTES_FREE( src->exchange );   
    AMQP_BYTES_FREE( src->routing_key );
    src = NULL ;
}


amqp_decimal_t notifymq_decimal_dup( const amqp_decimal_t src )
{
   amqp_decimal_t dest ;
   dest.decimals = src.decimals ;
   dest.value    = src.value ;
   return dest ;
}

void notifymq_table_dump( const amqp_table_t* t )
{
   int i ;
   amqp_table_entry_t* entry = NULL ;
   for( i = 0 ; i < t->num_entries ; i++ ){
      entry =  &t->entries[i] ;
      printf("_table_dump \"%c\" \n", entry->kind ); 
      //printf("_table_dump %.*s \n", entry->key.len , (char*)entry->key.bytes ); 
   } 
}


amqp_table_t notifymq_table_dup( const amqp_table_t src )
{
   amqp_table_t dest ;
   dest.num_entries = src.num_entries ;
   size_t len = src.num_entries * sizeof(amqp_table_entry_t) ;
   dest.entries = (amqp_table_entry_t*) malloc(len);
   int i ;
   for( i = 0 ; i < src.num_entries ; i++ ){
      amqp_table_entry_t* s =  &src.entries[i] ;
      amqp_table_entry_t* d =  &dest.entries[i] ;
      d->key  = amqp_bytes_malloc_dup( s->key );       
      d->kind = s->kind ;
      switch ( s->kind ){
          case 'S':
              d->value.bytes = amqp_bytes_malloc_dup( s->value.bytes ); 
              break ;
          case 'I':
              d->value.i32    = s->value.i32 ;
              break ;
          case 'D':
              d->value.decimal = notifymq_decimal_dup( s->value.decimal ); 
              break ;
          case 'T':
              d->value.u64    = s->value.u64 ;
              break ;
          case 'F':
              d->value.table  = notifymq_table_dup( s->value.table );
              break ;
      }
   }
   return dest ;
}

void notifymq_basic_properties_dump( const amqp_basic_properties_t* p )
{
    printf("basic_properties_dump \n" );
    if (p->_flags & AMQP_BASIC_CONTENT_TYPE_FLAG) 
        printf("\t.content_type\t\t%.*s\n", p->content_type.len, (char*)p->content_type.bytes );
    if (p->_flags & AMQP_BASIC_CONTENT_ENCODING_FLAG) 
        printf("\t.content_encoding\t\t%.*s\n", p->content_encoding.len, (char*)p->content_encoding.bytes );
    if (p->_flags & AMQP_BASIC_HEADERS_FLAG) 
        notifymq_table_dump( &p->headers );
    if (p->_flags & AMQP_BASIC_DELIVERY_MODE_FLAG) 
        printf("\t.delivery_mode\t\t%d\n", p->delivery_mode );
    if (p->_flags & AMQP_BASIC_PRIORITY_FLAG) 
        printf("\t.priority\t\t%d\n", p->priority );
    if (p->_flags & AMQP_BASIC_CORRELATION_ID_FLAG) 
        printf("\t.content_encoding\t\t%.*s\n", p->correlation_id.len, (char*)p->correlation_id.bytes );
    if (p->_flags & AMQP_BASIC_REPLY_TO_FLAG) 
        printf("\t.reply_to\t\t%.*s\n", p->reply_to.len, (char*)p->reply_to.bytes );
    if (p->_flags & AMQP_BASIC_EXPIRATION_FLAG) 
        printf("\t.expiration\t\t%.*s\n", p->expiration.len, (char*)p->expiration.bytes );
    if (p->_flags & AMQP_BASIC_MESSAGE_ID_FLAG) 
        printf("\t.message_id\t\t%.*s\n", p->message_id.len, (char*)p->message_id.bytes );
    if (p->_flags & AMQP_BASIC_TIMESTAMP_FLAG) 
        printf("\t.timestamp\t\t%lld\n", p->timestamp );
    if (p->_flags & AMQP_BASIC_TYPE_FLAG) 
        printf("\t.type\t\t%.*s\n", p->type.len, (char*)p->type.bytes );
    if (p->_flags & AMQP_BASIC_USER_ID_FLAG) 
        printf("\t.user_id\t\t%.*s\n", p->user_id.len, (char*)p->user_id.bytes );
    if (p->_flags & AMQP_BASIC_APP_ID_FLAG) 
        printf("\t.app_id\t\t%.*s\n", p->app_id.len, (char*)p->app_id.bytes );
    if (p->_flags & AMQP_BASIC_CLUSTER_ID_FLAG) 
        printf("\t.cluster_id\t\t%.*s\n", p->cluster_id.len, (char*)p->cluster_id.bytes );
}


amqp_basic_properties_t notifymq_basic_properties_dup( const amqp_basic_properties_t src )
{
    amqp_basic_properties_t dest ;
    dest._flags = src._flags ;

    if (src._flags & AMQP_BASIC_CONTENT_TYPE_FLAG) 
        dest.content_type     = amqp_bytes_malloc_dup( src.content_type );
    if (src._flags & AMQP_BASIC_CONTENT_ENCODING_FLAG) 
        dest.content_encoding = amqp_bytes_malloc_dup( src.content_encoding );

    if (src._flags & AMQP_BASIC_HEADERS_FLAG) 
        dest.headers          = notifymq_table_dup( src.headers );

    if (src._flags & AMQP_BASIC_DELIVERY_MODE_FLAG) 
        dest.delivery_mode    = src.delivery_mode ;
    if (src._flags & AMQP_BASIC_PRIORITY_FLAG) 
        dest.priority         = src.priority ;

    if (src._flags & AMQP_BASIC_CORRELATION_ID_FLAG) 
        dest.correlation_id   = amqp_bytes_malloc_dup( src.correlation_id ) ;
    if (src._flags & AMQP_BASIC_REPLY_TO_FLAG) 
        dest.reply_to         = amqp_bytes_malloc_dup( src.reply_to ) ;
    if (src._flags & AMQP_BASIC_EXPIRATION_FLAG) 
        dest.expiration       = amqp_bytes_malloc_dup( src.expiration ) ;
    if (src._flags & AMQP_BASIC_MESSAGE_ID_FLAG) 
        dest.message_id       = amqp_bytes_malloc_dup( src.message_id ) ;

    if (src._flags & AMQP_BASIC_TIMESTAMP_FLAG) 
        dest.timestamp        = src.timestamp ;

    if (src._flags & AMQP_BASIC_TYPE_FLAG) 
        dest.type             = amqp_bytes_malloc_dup( src.type ) ;
    if (src._flags & AMQP_BASIC_USER_ID_FLAG) 
        dest.user_id          = amqp_bytes_malloc_dup( src.user_id ) ;
    if (src._flags & AMQP_BASIC_APP_ID_FLAG) 
        dest.app_id           = amqp_bytes_malloc_dup( src.app_id ) ;
    if (src._flags & AMQP_BASIC_CLUSTER_ID_FLAG) 
        dest.cluster_id       = amqp_bytes_malloc_dup( src.cluster_id ) ;
    return dest ;
}


void notifymq_basic_properties_free( amqp_basic_properties_t* src )
{
    if (src->_flags & AMQP_BASIC_CONTENT_TYPE_FLAG) 
        AMQP_BYTES_FREE( src->content_type );
    if (src->_flags & AMQP_BASIC_CONTENT_ENCODING_FLAG) 
        AMQP_BYTES_FREE( src->content_encoding );

    //if (src->_flags & AMQP_BASIC_HEADERS_FLAG) 
    //    notifymq_table_free( src->headers );

    if (src->_flags & AMQP_BASIC_CORRELATION_ID_FLAG) 
        AMQP_BYTES_FREE( src->correlation_id ) ;
    if (src->_flags & AMQP_BASIC_REPLY_TO_FLAG) 
        AMQP_BYTES_FREE( src->reply_to ) ;
    if (src->_flags & AMQP_BASIC_EXPIRATION_FLAG) 
        AMQP_BYTES_FREE( src->expiration ) ;
    if (src->_flags & AMQP_BASIC_MESSAGE_ID_FLAG) 
        AMQP_BYTES_FREE( src->message_id ) ;

    if (src->_flags & AMQP_BASIC_TYPE_FLAG) 
        AMQP_BYTES_FREE( src->type ) ;
    if (src->_flags & AMQP_BASIC_USER_ID_FLAG) 
        AMQP_BYTES_FREE( src->user_id ) ;
    if (src->_flags & AMQP_BASIC_APP_ID_FLAG) 
        AMQP_BYTES_FREE( src->app_id ) ;
    if (src->_flags & AMQP_BASIC_CLUSTER_ID_FLAG) 
        AMQP_BYTES_FREE( src->cluster_id ) ;
    src = NULL ;
}








// property setters 


int notifymq_set_content_type( amqp_basic_properties_t* props , char const* v )
{
    if(!v) return 1 ;
    props->_flags |= AMQP_BASIC_CONTENT_TYPE_FLAG ;
    props->content_type = amqp_cstring_bytes(v);
    return 0 ;
}

char* notifymq_get_content_type( amqp_basic_properties_t* props )
{
    // unhealthy allocation 
    if (props->_flags & AMQP_BASIC_CONTENT_TYPE_FLAG) return notifymq_getstr_alloc( props->content_type );
    return NULL ;
}


int notifymq_set_content_encoding( amqp_basic_properties_t* props , char const* v )
{
    if(!v) return 1 ;
    props->_flags |= AMQP_BASIC_CONTENT_ENCODING_FLAG ;
    props->content_encoding = amqp_cstring_bytes(v);
    return 0 ;
}

int notifymq_set_correlation_id( amqp_basic_properties_t* props , char const* v )
{
    if(!v) return 1 ;
    props->_flags |= AMQP_BASIC_CORRELATION_ID_FLAG ;
    props->correlation_id = amqp_cstring_bytes(v);
    return 0 ;
}

int notifymq_set_reply_to( amqp_basic_properties_t* props , char const* v )
{
    if(!v) return 1 ;
    props->_flags |= AMQP_BASIC_REPLY_TO_FLAG ;
    props->reply_to = amqp_cstring_bytes(v);
    return 0 ;
}

int notifymq_set_expiration( amqp_basic_properties_t* props , char const* v )
{
    if(!v) return 1 ;
    props->_flags |= AMQP_BASIC_EXPIRATION_FLAG ;
    props->expiration = amqp_cstring_bytes(v);
    return 0 ;
}

int notifymq_set_message_id( amqp_basic_properties_t* props , char const* v )
{
    if(!v) return 1 ;
    props->_flags |= AMQP_BASIC_MESSAGE_ID_FLAG ;
    props->message_id = amqp_cstring_bytes(v);
    return 0 ;
}

int notifymq_set_type( amqp_basic_properties_t* props , char const* v )
{
    if(!v) return 1 ;
    props->_flags |= AMQP_BASIC_TYPE_FLAG ;
    props->type = amqp_cstring_bytes(v);
    return 0 ;
}

int notifymq_set_user_id( amqp_basic_properties_t* props , char const* v )
{
    if(!v) return 1 ;
    props->_flags |= AMQP_BASIC_USER_ID_FLAG ;
    props->user_id = amqp_cstring_bytes(v);
    return 0 ;
}

int notifymq_set_app_id( amqp_basic_properties_t* props , char const* v )
{
    if(!v) return 1 ;
    props->_flags |= AMQP_BASIC_APP_ID_FLAG ;
    props->app_id = amqp_cstring_bytes(v);
    return 0 ;
}

int notifymq_set_cluster_id( amqp_basic_properties_t* props , char const* v )
{
    if(!v) return 1 ;
    props->_flags |= AMQP_BASIC_CLUSTER_ID_FLAG ;
    props->cluster_id = amqp_cstring_bytes(v);
    return 0 ;
}

int notifymq_set_delivery_mode( amqp_basic_properties_t* props, uint8_t v )
{
    props->_flags |= AMQP_BASIC_DELIVERY_MODE_FLAG ;
    props->delivery_mode = v ;
    return 0 ;
}

int notifymq_set_priority( amqp_basic_properties_t* props, uint8_t v )
{
    props->_flags |= AMQP_BASIC_PRIORITY_FLAG ;
    props->priority = v ;
    return 0 ;
}

int notifymq_set_timestamp( amqp_basic_properties_t* props, uint64_t v )
{
    props->_flags |= AMQP_BASIC_TIMESTAMP_FLAG ;
    props->timestamp = v ;
    return 0 ;
}







int notifymq_sendbytes( char const*  exchange , char const* routingkey , void* msgbytes , size_t msglen )
{ 
   // http://hg.rabbitmq.com/rabbitmq-c/file/712d3c55f2b5/examples/amqp_producer.c
    amqp_basic_properties_t props;
    props._flags = 0 ;
    notifymq_set_content_type(      &props , "application/data" );

    char* s ;
    s = notifymq_get_content_type( &props );
    printf("sendbytes \"%s\" \n", s );


    notifymq_set_content_encoding(  &props , "binary" );
    notifymq_set_delivery_mode(     &props , 2 );           // persistent delivery mode

    die_on_error(amqp_basic_publish(conn,
				    1,
				    amqp_cstring_bytes(exchange),
				    amqp_cstring_bytes(routingkey),
				    0,
				    0,
				    &props,
				    (amqp_bytes_t){.len = msglen, .bytes = msgbytes }),
		 "Publishing");
    return EXIT_SUCCESS ;
}









int notifymq_sendstring( char const*  exchange , char const* routingkey , char const* messagebody )
{ 
   // http://hg.rabbitmq.com/rabbitmq-c/file/712d3c55f2b5/examples/amqp_sendstring.c
    amqp_basic_properties_t props;
    props._flags = 0 ;
    notifymq_set_content_type(      &props , "text/plain" );
    //notifymq_set_content_encoding(  &props , "test.encoding" );
    //notifymq_set_headers(         &props  , "??" ) ;
    notifymq_set_delivery_mode(    &props , 2 );           // persistent delivery mode
    notifymq_set_priority(         &props , 1 ); 
    notifymq_set_correlation_id(   &props , "test.correlation_id" ); 
    notifymq_set_reply_to(         &props , "test.reply_to" ); 
    notifymq_set_expiration(       &props , "test.expiration" ); 
    notifymq_set_message_id(       &props , "test.message_id" ); 
    notifymq_set_timestamp(        &props ,  (uint64_t)101 ); 
    notifymq_set_type(             &props , "test.type" ); 
    notifymq_set_user_id(          &props , "test.user_id" ); 
    notifymq_set_app_id(           &props , "test.app_id" ); 
    notifymq_set_cluster_id(       &props , "test.cluster_id" ); 


    die_on_error(amqp_basic_publish(conn,
				    1,
				    amqp_cstring_bytes(exchange),
				    amqp_cstring_bytes(routingkey),
				    0,
				    0,
				    &props,
				    amqp_cstring_bytes(messagebody)),
		 "Publishing");
    return EXIT_SUCCESS ;
}

int notifymq_cleanup()
{
    die_on_amqp_error(amqp_channel_close(conn, 1, AMQP_REPLY_SUCCESS), "Closing channel");
    die_on_amqp_error(amqp_connection_close(conn, AMQP_REPLY_SUCCESS), "Closing connection");
    amqp_destroy_connection(conn);
    die_on_error(close(sockfd), "Closing socket");
    private_cleanup();
    return EXIT_SUCCESS ;
}



int notifymq_basic_consume( char const* queue , receiver_t handlebytes , void* arg ) 
{
   // based on rabbitmq-c/examples/amqp_listen.c

  int dbg = notifymq_dbg ; 
  amqp_boolean_t no_local   = 0 ; 
  amqp_boolean_t no_ack     = 1 ;
  amqp_boolean_t exclusive  = 0 ;

  long long start_time = now_microseconds()  ;
  //long long live_time = 1000000 ;  
  long long live_time = 0 ;  
  long long cycle_time ;

  amqp_basic_consume(conn, 1, amqp_cstring_bytes(queue) , 
                           AMQP_EMPTY_BYTES , no_local, no_ack, exclusive );
  die_on_amqp_error(amqp_rpc_reply, "Consuming");

  {
    amqp_frame_t frame;
    int result;

    amqp_basic_deliver_t *d;
    amqp_basic_properties_t *p;
    size_t body_target;
    size_t body_received;
      
    notifymq_props_t props ;

    long long cycle ;
    if(dbg>0) printf("notifymq_basic_consume : Starting wait loop \n");
    while (1) {

      cycle++ ;

      //if( cycle % 1000 == 0 ){
      //    printf("cycle %lld \n" , cycle );
      //}

      cycle_time = now_microseconds();
      //if (live_time != 0 && cycle_time > start_time + live_time ){
      //   printf("Time to die after %lld cycles \n", cycle );
      //   break ; 
      //}

      amqp_maybe_release_buffers(conn);
      result = amqp_simple_wait_frame(conn, &frame);   // the wait happens here 
      if(dbg>1) printf("Result %d\n", result);
      if (result <= 0)
	break;

      if(dbg > 2 ){
          printf("Cycles %lld\n", cycle);
          printf("start_time %lld \n", start_time );
          printf("live_time  %lld \n", live_time );
          printf("cycle_time %lld \n", cycle_time );
      } 

      if(dbg>0) printf("Frame type %d, channel %d\n", frame.frame_type, frame.channel);
      if (frame.frame_type != AMQP_FRAME_METHOD)
	continue;

      if(dbg>0) printf("Method %s\n", amqp_method_name(frame.payload.method.id));
      if (frame.payload.method.id != AMQP_BASIC_DELIVER_METHOD)
	continue;

      d = (amqp_basic_deliver_t *) frame.payload.method.decoded;
      if(dbg>0) printf("Delivery %u, exchange %.*s routingkey %.*s consumertag %.*s\n",
	     (unsigned) d->delivery_tag,
	     (int) d->exchange.len, (char *) d->exchange.bytes,
	     (int) d->routing_key.len, (char *) d->routing_key.bytes,
	     (int) d->consumer_tag.len, (char *) d->consumer_tag.bytes);

      result = amqp_simple_wait_frame(conn, &frame);
      if (result <= 0)
	break;

      if (frame.frame_type != AMQP_FRAME_HEADER) {
	fprintf(stderr, "Expected header!");
	abort();
      }
      p = (amqp_basic_properties_t *) frame.payload.properties.decoded;


      props.content_type.len   = 0 ;
      props.content_type.bytes = NULL ;
      if (p->_flags & AMQP_BASIC_CONTENT_TYPE_FLAG) {
	if(dbg>0) printf("Content-type: %.*s\n",
	       (int) p->content_type.len, (char *) p->content_type.bytes);
         // collect metadata for the handlebytes call
         //props.content_type = amqp_bytes_malloc_dup( p->content_type );
         props.content_type.len   = p->content_type.len ;
         props.content_type.bytes = p->content_type.bytes ;
      }


 
      props.content_encoding.len   = 0 ;
      props.content_encoding.bytes = NULL ; 
      if (p->_flags & AMQP_BASIC_CONTENT_ENCODING_FLAG) {
	if(dbg>0) printf("Content-encoding: %.*s\n",
	       (int) p->content_encoding.len, (char *) p->content_encoding.bytes);
         // collect metadata for the handlebytes call
         //props.content_encoding = amqp_bytes_malloc_dup( p->content_encoding );
         props.content_encoding.len   = p->content_encoding.len ;
         props.content_encoding.bytes = p->content_encoding.bytes ;
      }


      if(dbg>0) printf("----\n");

      body_target = frame.payload.properties.body_size;
      body_received = 0;

      while (body_received < body_target) {
	result = amqp_simple_wait_frame(conn, &frame);
	if (result <= 0)
	  break;

	if (frame.frame_type != AMQP_FRAME_BODY) {
	  fprintf(stderr, "Expected body!");
	  abort();
	}	  

	body_received += frame.payload.body_fragment.len;
	assert(body_received <= body_target);

        if(dbg>1) amqp_dump(frame.payload.body_fragment.bytes, frame.payload.body_fragment.len);
      }


      if (body_received != body_target) {
	/* Can only happen when amqp_simple_wait_frame returns <= 0 */
	/* We break here to close the connection */
	break;
      }

        
      if(dbg>0) fprintf(stderr, "notifymq_basic_consume : invoking the receiver \n");
      
      // perhaps should dupe first ?
      //amqp_bytes_t dupe = amqp_bytes_malloc_dup( frame.payload.body_fragment );
      //handlebytes( arg, dupe.bytes, dupe.len);

      //    
      //  p = (amqp_basic_properties_t *) frame.payload.properties.decoded;
      //  d = (amqp_basic_deliver_t *)    frame.payload.method.decoded;
      
      notifymq_basic_collect( &frame.payload.body_fragment , d , p );
      handlebytes( arg , frame.payload.body_fragment.bytes, frame.payload.body_fragment.len , props );
      

    }
  }
  return EXIT_SUCCESS ;
}







