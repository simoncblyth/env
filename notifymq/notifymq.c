

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


int notifymq_init()
{
    int rc = private_init();
    if(rc != EXIT_SUCCESS) return rc ;
   
    char const* hostname = private_lookup("AMQP_SERVER");
    int port = atoi(private_lookup("AMQP_PORT"));
    char const* user = private_lookup("AMQP_USER");
    char const* password = private_lookup("AMQP_PASSWORD");
    char const* vhost = private_lookup("AMQP_VHOST");

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


int notifymq_sendbytes( char const*  exchange , char const* routingkey , void* msgbytes , size_t msglen )
{ 
   // http://hg.rabbitmq.com/rabbitmq-c/file/712d3c55f2b5/examples/amqp_producer.c
    amqp_basic_properties_t props;
    props._flags = AMQP_BASIC_CONTENT_TYPE_FLAG | AMQP_BASIC_DELIVERY_MODE_FLAG | AMQP_BASIC_CONTENT_ENCODING_FLAG ;
    props.content_type = amqp_cstring_bytes("application/data");
    props.content_encoding = amqp_cstring_bytes("binary");
    props.delivery_mode = 2; // persistent delivery mode
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
    props._flags = AMQP_BASIC_CONTENT_TYPE_FLAG | AMQP_BASIC_DELIVERY_MODE_FLAG;
    props.content_type = amqp_cstring_bytes("text/plain");
    props.delivery_mode = 2; // persistent delivery mode
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

int notifymq_basic_consume( char const* queue ) 
{
   // based on rabbitmq-c/examples/amqp_listen.c

  amqp_boolean_t no_local   = 0 ; 
  amqp_boolean_t no_ack     = 1 ;
  amqp_boolean_t exclusive  = 0 ;

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

    while (1) {
      amqp_maybe_release_buffers(conn);
      result = amqp_simple_wait_frame(conn, &frame);
      printf("Result %d\n", result);
      if (result <= 0)
	break;

      printf("Frame type %d, channel %d\n", frame.frame_type, frame.channel);
      if (frame.frame_type != AMQP_FRAME_METHOD)
	continue;

      printf("Method %s\n", amqp_method_name(frame.payload.method.id));
      if (frame.payload.method.id != AMQP_BASIC_DELIVER_METHOD)
	continue;

      d = (amqp_basic_deliver_t *) frame.payload.method.decoded;
      printf("Delivery %u, exchange %.*s routingkey %.*s\n",
	     (unsigned) d->delivery_tag,
	     (int) d->exchange.len, (char *) d->exchange.bytes,
	     (int) d->routing_key.len, (char *) d->routing_key.bytes);

      result = amqp_simple_wait_frame(conn, &frame);
      if (result <= 0)
	break;

      if (frame.frame_type != AMQP_FRAME_HEADER) {
	fprintf(stderr, "Expected header!");
	abort();
      }
      p = (amqp_basic_properties_t *) frame.payload.properties.decoded;
      if (p->_flags & AMQP_BASIC_CONTENT_TYPE_FLAG) {
	printf("Content-type: %.*s\n",
	       (int) p->content_type.len, (char *) p->content_type.bytes);
      }
      printf("----\n");

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

	amqp_dump(frame.payload.body_fragment.bytes,
		  frame.payload.body_fragment.len);
      }

      if (body_received != body_target) {
	/* Can only happen when amqp_simple_wait_frame returns <= 0 */
	/* We break here to close the connection */
	break;
      }
    }
  }
  return EXIT_SUCCESS ;
}
