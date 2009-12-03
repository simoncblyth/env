#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>

#include <amqp.h>
#include <amqp_framing.h>

#include "notifymq.h"
#include "notifymq_utils.h"

extern void amqp_dump(void const *buffer, size_t len);


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

void notifymq_basic_msg_dump( const notifymq_basic_msg_t* msg , int verbosity , const char* label )
{
    if( msg == NULL ){
       printf("notifymq_msg_dump %s ERROR null msg \n", label );
       return ;
    }
    printf("notifymq_msg_dump %s .index %lld .key \"%s\" verbosity %d \n", label , msg->index , msg->key, verbosity  );
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


void notifymq_basic_deliver_dump( const amqp_basic_deliver_t* d )
{
    printf("basic_deliver_dump delivery_tag %lld redelivered %d \n", d->delivery_tag, d->redelivered );
    printf("\t.consumer_tag\t\t%.*s\n",  (int)d->consumer_tag.len, (char*)d->consumer_tag.bytes );
    printf("\t.exchange\t\t%.*s\n",      (int)d->exchange.len,     (char*)d->exchange.bytes );
    printf("\t.routing_key\t\t%.*s\n",   (int)d->routing_key.len,  (char*)d->routing_key.bytes );
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
        printf("\t.content_type\t\t%.*s\n", (int)p->content_type.len, (char*)p->content_type.bytes );
    if (p->_flags & AMQP_BASIC_CONTENT_ENCODING_FLAG) 
        printf("\t.content_encoding\t\t%.*s\n", (int)p->content_encoding.len, (char*)p->content_encoding.bytes );
    if (p->_flags & AMQP_BASIC_HEADERS_FLAG) 
        notifymq_table_dump( &p->headers );
    if (p->_flags & AMQP_BASIC_DELIVERY_MODE_FLAG) 
        printf("\t.delivery_mode\t\t%d\n", p->delivery_mode );
    if (p->_flags & AMQP_BASIC_PRIORITY_FLAG) 
        printf("\t.priority\t\t%d\n", p->priority );
    if (p->_flags & AMQP_BASIC_CORRELATION_ID_FLAG) 
        printf("\t.content_encoding\t\t%.*s\n", (int)p->correlation_id.len, (char*)p->correlation_id.bytes );
    if (p->_flags & AMQP_BASIC_REPLY_TO_FLAG) 
        printf("\t.reply_to\t\t%.*s\n", (int)p->reply_to.len, (char*)p->reply_to.bytes );
    if (p->_flags & AMQP_BASIC_EXPIRATION_FLAG) 
        printf("\t.expiration\t\t%.*s\n", (int)p->expiration.len, (char*)p->expiration.bytes );
    if (p->_flags & AMQP_BASIC_MESSAGE_ID_FLAG) 
        printf("\t.message_id\t\t%.*s\n", (int)p->message_id.len, (char*)p->message_id.bytes );
    if (p->_flags & AMQP_BASIC_TIMESTAMP_FLAG) 
        printf("\t.timestamp\t\t%lld\n", p->timestamp );
    if (p->_flags & AMQP_BASIC_TYPE_FLAG) 
        printf("\t.type\t\t%.*s\n", (int)p->type.len, (char*)p->type.bytes );
    if (p->_flags & AMQP_BASIC_USER_ID_FLAG) 
        printf("\t.user_id\t\t%.*s\n", (int)p->user_id.len, (char*)p->user_id.bytes );
    if (p->_flags & AMQP_BASIC_APP_ID_FLAG) 
        printf("\t.app_id\t\t%.*s\n", (int)p->app_id.len, (char*)p->app_id.bytes );
    if (p->_flags & AMQP_BASIC_CLUSTER_ID_FLAG) 
        printf("\t.cluster_id\t\t%.*s\n", (int)p->cluster_id.len, (char*)p->cluster_id.bytes );
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






// getter ... with unhealthy allocation 

const char* notifymq_props_get_content_type( amqp_basic_properties_t* props )
{
    if (props->_flags & AMQP_BASIC_CONTENT_TYPE_FLAG) return notifymq_getstr_alloc( props->content_type );
    return NULL ;
}

const char* notifymq_props_get_content_encoding( amqp_basic_properties_t* props )
{
    if (props->_flags & AMQP_BASIC_CONTENT_ENCODING_FLAG) return notifymq_getstr_alloc( props->content_encoding );
    return NULL ;
} 




// property setters 

int notifymq_set_content_type( amqp_basic_properties_t* props , char const* v )
{
    if(!v) return 1 ;
    props->_flags |= AMQP_BASIC_CONTENT_TYPE_FLAG ;
    props->content_type = amqp_cstring_bytes(v);
    return 0 ;
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


