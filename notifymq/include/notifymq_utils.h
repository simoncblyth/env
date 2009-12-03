#ifndef notifymq_utils_h
#define notifymq_utils_h

#include <stdint.h>
#include <amqp.h>
#include <amqp_framing.h>
#include "notifymq.h"

#ifdef __cplusplus
extern "C"
{
#endif

char* notifymq_getstr_alloc( amqp_bytes_t b ) ;
int notifymq_getstr( amqp_bytes_t b , char* buf , size_t max  ) ;

// duplicators 
amqp_basic_deliver_t    notifymq_basic_deliver_dup( const amqp_basic_deliver_t src );
amqp_decimal_t          notifymq_decimal_dup( const amqp_decimal_t src );
amqp_table_t            notifymq_table_dup( const amqp_table_t src );
notifymq_basic_msg_t*   notifymq_basic_msg_dup( uint64_t index , amqp_bytes_t* body ,  amqp_basic_deliver_t* deliver , amqp_basic_properties_t* props );
amqp_basic_properties_t notifymq_basic_properties_dup( const amqp_basic_properties_t src );

// dumpers 
void notifymq_basic_deliver_dump( const amqp_basic_deliver_t* d );
void notifymq_table_dump( const amqp_table_t* t );
void notifymq_basic_msg_dump( const notifymq_basic_msg_t* msg, int verbosity , const char* label  );
void notifymq_basic_properties_dump( const amqp_basic_properties_t* p );

// deallocaters
void notifymq_basic_msg_free( notifymq_basic_msg_t* msg );
void notifymq_basic_deliver_free( amqp_basic_deliver_t* src );
void notifymq_basic_properties_free( amqp_basic_properties_t* src );

// getter ... with allocation 
const char* notifymq_props_get_content_type(     amqp_basic_properties_t* props );
const char* notifymq_props_get_content_encoding( amqp_basic_properties_t* props );

// property setters 
int notifymq_set_content_type( amqp_basic_properties_t* props , char const* v );
int notifymq_set_content_encoding( amqp_basic_properties_t* props , char const* v );
int notifymq_set_correlation_id( amqp_basic_properties_t* props , char const* v );
int notifymq_set_reply_to( amqp_basic_properties_t* props , char const* v );
int notifymq_set_expiration( amqp_basic_properties_t* props , char const* v );
int notifymq_set_message_id( amqp_basic_properties_t* props , char const* v );
int notifymq_set_type( amqp_basic_properties_t* props , char const* v );
int notifymq_set_user_id( amqp_basic_properties_t* props , char const* v );
int notifymq_set_app_id( amqp_basic_properties_t* props , char const* v );
int notifymq_set_cluster_id( amqp_basic_properties_t* props , char const* v );
int notifymq_set_delivery_mode( amqp_basic_properties_t* props, uint8_t v );
int notifymq_set_priority( amqp_basic_properties_t* props, uint8_t v );
int notifymq_set_timestamp( amqp_basic_properties_t* props, uint64_t v );

#ifdef __cplusplus
}
#endif
#endif
