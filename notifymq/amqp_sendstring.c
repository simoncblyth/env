#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <stdint.h>
#include <amqp.h>
#include <amqp_framing.h>

#include <unistd.h>
#include "private.h"
#include "example_utils.h"


int notifymq_init();
int notifymq_cleanup();
int notifymq_sendstring( char const*  exchange , char const* routingkey , char const* messagebody );


int sockfd ;
amqp_connection_state_t conn ;




int notifymq_init()
{
    private_init();
    char const* hostname = private_lookup("AMQP_SERVER");
    int port = atoi(private_lookup("AMQP_PORT"));
    char const* user = private_lookup("AMQP_USER");
    char const* password = private_lookup("AMQP_PASSWORD");
    char const* vhost = private_lookup("AMQP_VHOST");

    die_on_error(sockfd = amqp_open_socket(hostname, port), "Opening socket");
    conn = amqp_new_connection();
    amqp_set_sockfd(conn, sockfd);
    die_on_amqp_error(amqp_login(conn, vhost , 0, 131072, 0, AMQP_SASL_METHOD_PLAIN, user, password), "Logging in");
    amqp_channel_open(conn, 1);
    die_on_amqp_error(amqp_rpc_reply, "Opening channel");
    return EXIT_SUCCESS ;
}

int notifymq_sendstring( char const*  exchange , char const* routingkey , char const* messagebody )
{ 
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


int main(int argc, char const * const *argv) {
   if (argc < 4) {
      fprintf(stderr, "Usage: amqp_sendstring exchange routingkey messagebody\n");
      return 1;
   }
   notifymq_init();
   char const* exchange = argv[1];
   char const* routingkey = argv[2];
   char const* messagebody = argv[3];
   notifymq_sendstring( exchange , routingkey , messagebody );
   notifymq_cleanup();
   return 0;
}
