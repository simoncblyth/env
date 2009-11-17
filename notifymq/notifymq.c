

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <stdint.h>
#include <amqp.h>
#include <amqp_framing.h>

#include <unistd.h>
#include "private.h"
#include "example_utils.h"

#include "notifymq.h"

static int sockfd ;
static amqp_connection_state_t conn ;


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


int notifymq_exchange_declare( char const* exchange , char const* exchangetype )
{
    amqp_exchange_declare(conn, 
                              1, 
                              amqp_cstring_bytes(exchange), 
                              amqp_cstring_bytes(exchangetype),
     			      0, 0, 0, AMQP_EMPTY_TABLE);
    die_on_amqp_error(amqp_rpc_reply, "Declaring exchange");
    return EXIT_SUCCESS ;
}


int notifymq_sendbytes( char const*  exchange , char const* routingkey , void* msgbytes , size_t msglen )
{ 
   // http://hg.rabbitmq.com/rabbitmq-c/file/712d3c55f2b5/examples/amqp_producer.c
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


