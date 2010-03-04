

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
#include "notifymq_utils.h"
#include "notifymq_collection.h"

#include <glib.h>


static int sockfd ;
static amqp_connection_state_t conn ;
extern void amqp_dump(void const *buffer, size_t len);

int notifymq_dbg = 0 ;


static uint64_t notifymq_msg_index = 0 ; 
static GThread* notifymq_monitor_thread = NULL;

const char* notifymq_get_content_type( notifymq_basic_msg_t* msg )
{
    return  msg == NULL ? NULL : notifymq_props_get_content_type( &(msg->properties) )  ;    
}

const char* notifymq_get_content_encoding( notifymq_basic_msg_t* msg )
{
    return  msg == NULL ? NULL : notifymq_props_get_content_encoding( &(msg->properties) )  ;    
}


int notifymq_basic_collect( amqp_bytes_t* body ,  amqp_basic_deliver_t* deliver , amqp_basic_properties_t* props )
{
	// bundle message body and delivery metadata and properties into struct notifymq_basic_msg_t and add to collection 
	
    // dynamically allocated duplication of the message pulled off the wire ... 
    // the bytes of the frame are shortly deallocated so must do the duping 
    // for them to survive into the glib data structure
    notifymq_basic_msg_t* msg = notifymq_basic_msg_dup( notifymq_msg_index , body , deliver , props );
    notifymq_msg_index++ ;     // global received message index within this execution

    if( notifymq_dbg > 2 )  
        notifymq_basic_msg_dump( msg , 3 , "_basic_collect");

    // data structure just keeps the msg pointer ... no copying 
    // collection assumes ownership of the msg data and will free it once the number of messages exceeds the maximum
    notifymq_collection_add( msg );
    
    if( notifymq_dbg > 2 )  
        notifymq_collection_dump();

    return EXIT_SUCCESS ;
}



int notifymq_init()
{

    if (!g_thread_supported ()) g_thread_init (NULL);

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





    notifymq_collection_init();

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



int notifymq_sendbytes( char const*  exchange , char const* routingkey , void* msgbytes , size_t msglen )
{ 
   // http://hg.rabbitmq.com/rabbitmq-c/file/712d3c55f2b5/examples/amqp_producer.c
    amqp_basic_properties_t props;
    props._flags = 0 ;
    notifymq_set_content_type(      &props , "application/data" );
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

    notifymq_collection_cleanup();
    private_cleanup();
    return EXIT_SUCCESS ;
}


gpointer notifymq_monitor_thread_(gpointer data )
{
	// waits on message bytes, invoking notifymq_basic_collect once complete messages are received
    char const* queue = (char const*)data ; 
    if( notifymq_dbg > 0 )
        printf("notifymq_monitor_thread_ starting for queue \"%s\"\n", queue );
    notifymq_basic_consume( queue );
    return NULL ;
}

int notifymq_basic_consume_async( char const* queue ) 
{
	// spin off the message queue monitor thread  
    gboolean joinable = 0 ;
    notifymq_monitor_thread = g_thread_create((GThreadFunc)notifymq_monitor_thread_ , (gpointer)queue , joinable, NULL);
    return EXIT_SUCCESS ;
}


int notifymq_basic_consume( char const* queue )  //  , receiver_t handlebytes , void* arg ) 
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

      if (p->_flags & AMQP_BASIC_CONTENT_TYPE_FLAG) {
	if(dbg>0) printf("Content-type: %.*s\n",
	       (int) p->content_type.len, (char *) p->content_type.bytes);
      }

      if (p->_flags & AMQP_BASIC_CONTENT_ENCODING_FLAG) {
	if(dbg>0) printf("Content-encoding: %.*s\n",
	       (int) p->content_encoding.len, (char *) p->content_encoding.bytes);
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
      
     
      notifymq_basic_collect( &frame.payload.body_fragment , d , p );
      //handlebytes( arg , frame.payload.body_fragment.bytes, frame.payload.body_fragment.len , props );
      
      // as frame goes out of scope the body/delivery/property bytes are deallocated ... 
    }
  }
  return EXIT_SUCCESS ;
}







