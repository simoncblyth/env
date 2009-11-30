

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

#include "notifymq.h"

static int sockfd ;
static amqp_connection_state_t conn ;

extern void amqp_dump(void const *buffer, size_t len);

static int notifymq_dbg = 0 ; 

int notifymq_getstr( amqp_bytes_t b , char* buf , size_t max  ) {
    if(b.len > max || b.bytes == 0) return EXIT_FAILURE ;
    return snprintf( buf , max , "%s" , (char*)b.bytes );
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



// property setters 


int notifymq_set_content_type( amqp_basic_properties_t* props , char const* v )
{
    if(!v) return 1 ;
    props->_flags |= AMQP_BASIC_CONTENT_TYPE_FLAG ;
    props->content_type = amqp_cstring_bytes(v);
    return 0 ;
}

int notifymq_get_content_type( amqp_basic_properties_t* props, char* buf , size_t max )
{
    if (props->_flags & AMQP_BASIC_CONTENT_TYPE_FLAG){
        notifymq_getstr( props->content_type, buf , max  );
        printf("get_content_type %s \n", buf );
    }
    return EXIT_FAILURE ;
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

    const size_t max = 30 ;  
    char s[max] ;
    notifymq_get_content_type(      &props , s , max  );
    printf("sendbytes %s \n", s );


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

    const size_t max = 30 ;
    char s[max] ;
    notifymq_get_content_type(      &props , s , max );
    printf("sendstring %s \n", s );




    //notifymq_set_content_encoding( &props , "?" );
    //notifymq_set_headers(         &props  , "??" ) ;
    notifymq_set_delivery_mode(     &props , 2 );           // persistent delivery mode
    //notifymq_set_priority(          &props , 1 ); 
    notifymq_set_correlation_id(   &props , "test.correlation_id" ); 
    notifymq_set_reply_to(         &props , "test.reply_to" ); 
    notifymq_set_expiration(       &props , "test.expiration" ); 
    notifymq_set_message_id(       &props , "test.message_id" ); 
    //notifymq_set_timestamp(      &props ,  (uint64_t)101 ); 
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


      const size_t max = 40 ;
      char s[max] ;
      notifymq_get_content_type( p , s , max );
      printf("notifymq_get_content_type: %s\n", s );


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


      handlebytes( arg , frame.payload.body_fragment.bytes, frame.payload.body_fragment.len , props );
      

    }
  }
  return EXIT_SUCCESS ;
}
