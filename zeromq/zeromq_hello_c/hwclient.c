/*

  name=hwclient && cc -I$VIRTUAL_ENV/include -c $name.c && cc -L$VIRTUAL_ENV/lib -lzmq $name.o -o /tmp/$name && rm $name.o 


*/
//  Hello World client   http://zguide.zeromq.org/c:hwclient
#include <zmq.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <assert.h>

int main (void)
{

    char* CONFIG = getenv("HELLO_CLIENT_CONFIG") ; 
    if( !CONFIG )
    {
        printf("HELLO_CLIENT_CONFIG not defined \n");
        return 1; 
    } 

    void* context = zmq_ctx_new ();
    void* requester = zmq_socket (context, ZMQ_REQ);
    int rc = zmq_connect (requester, CONFIG );
    assert (rc == 0);


    int request_nbr;
    for (request_nbr = 0; request_nbr != 10; request_nbr++) {
        char buffer [10];
        sprintf( buffer, "%s%d", "hello", request_nbr );

        printf ("zmq_send [%s] ...", buffer);
        zmq_send (requester, buffer, 10, 0);
        printf ("... after zmq_send [%s]\n", buffer);

        printf ("zmq_recv ...");
        zmq_recv (requester, buffer, 10, 0);
        printf ("... after zmq_recv [%s]\n", buffer );
    }
    zmq_close (requester);
    zmq_ctx_destroy (context);
    return 0;
}
