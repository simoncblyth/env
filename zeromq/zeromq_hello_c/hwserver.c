/*
  name=hwserver && cc -I$VIRTUAL_ENV/include -c $name.c && cc -L$VIRTUAL_ENV/lib -lzmq $name.o -o /tmp/$name && rm $name.o 

  

*/


#include <zmq.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <assert.h>

int main (void)
{

    char* CONFIG = getenv("HELLO_SERVER_CONFIG") ; 
    if( !CONFIG )
    {
        printf("HELLO_SERVER_CONFIG not defined \n");
        return 1; 
    } 

    void* context = zmq_ctx_new ();
    void* responder = zmq_socket (context, ZMQ_REP); 
    int rc = zmq_bind (responder, CONFIG );    // the server "zmq_bind"s the client "zmq_connect"s  ?
    assert (rc == 0); // maybe another server running OR port occupied 

    while (1) {
        char buffer [10];

        printf ("zmq_recv ...");
        zmq_recv (responder, buffer, 10, 0);
        printf ("... after zmq_recv [%s]\n", buffer);

        sleep(3);          //  Do some 'work'

        printf ("zmq_send ...");
        zmq_send (responder, buffer, 10, 0);
        printf ("... after zmq_send [%s]\n", buffer);
    }

    zmq_close( responder );
    zmq_ctx_destroy( context );

    return 0;
}
