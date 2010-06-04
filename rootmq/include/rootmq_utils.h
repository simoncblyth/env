#ifndef rootmq_utils_h
#define rootmq_utils_h

#include <stdint.h>
#include <amqp.h>
#include <amqp_framing.h>
#include "rootmq.h"

#ifdef __cplusplus
extern "C"
{
#endif

static char* rootmq_getstr_alloc( amqp_bytes_t b ) ;
int rootmq_getstr( amqp_bytes_t b , char* buf , size_t max  ) ;

// duplicators 
amqp_basic_deliver_t    rootmq_basic_deliver_dup( const amqp_basic_deliver_t src );
amqp_decimal_t          rootmq_decimal_dup( const amqp_decimal_t src );
amqp_table_t            rootmq_table_dup( const amqp_table_t src );
rootmq_basic_msg_t*   rootmq_basic_msg_dup( uint64_t index , amqp_bytes_t* body ,  amqp_basic_deliver_t* deliver , amqp_basic_properties_t* props );
amqp_basic_properties_t rootmq_basic_properties_dup( const amqp_basic_properties_t src );

// dumpers 
void rootmq_basic_deliver_dump( const amqp_basic_deliver_t* d );
void rootmq_table_dump( const amqp_table_t* t );
void rootmq_basic_msg_dump( const rootmq_basic_msg_t* msg, int verbosity , const char* label  );
void rootmq_basic_properties_dump( const amqp_basic_properties_t* p );

// deallocaters
void rootmq_basic_msg_free( rootmq_basic_msg_t* msg );
void rootmq_basic_deliver_free( amqp_basic_deliver_t* src );
void rootmq_basic_properties_free( amqp_basic_properties_t* src );

// getter ... with allocation 
const char* rootmq_props_get_content_type(     amqp_basic_properties_t* props );
const char* rootmq_props_get_content_encoding( amqp_basic_properties_t* props );

// property setters 
int rootmq_set_content_type( amqp_basic_properties_t* props , char const* v );
int rootmq_set_content_encoding( amqp_basic_properties_t* props , char const* v );
int rootmq_set_correlation_id( amqp_basic_properties_t* props , char const* v );
int rootmq_set_reply_to( amqp_basic_properties_t* props , char const* v );
int rootmq_set_expiration( amqp_basic_properties_t* props , char const* v );
int rootmq_set_message_id( amqp_basic_properties_t* props , char const* v );
int rootmq_set_type( amqp_basic_properties_t* props , char const* v );
int rootmq_set_user_id( amqp_basic_properties_t* props , char const* v );
int rootmq_set_app_id( amqp_basic_properties_t* props , char const* v );
int rootmq_set_cluster_id( amqp_basic_properties_t* props , char const* v );
int rootmq_set_delivery_mode( amqp_basic_properties_t* props, uint8_t v );
int rootmq_set_priority( amqp_basic_properties_t* props, uint8_t v );
int rootmq_set_timestamp( amqp_basic_properties_t* props, uint64_t v );

#ifdef __cplusplus
}
#endif
#endif
