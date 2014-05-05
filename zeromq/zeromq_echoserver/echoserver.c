//name=echoserver && cc -I$ZEROMQ_PREFIX/include -c $name.c && cc -L$ZEROMQ_PREFIX/lib -lzmq $name.o -o /tmp/$name && rm $name.o 

#include <zmq.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <assert.h>

void do_bind( void* socket , char* envvar )
{
    char* config = getenv(envvar) ; 
    if( !config )
    {
        printf("envvar %s not defined \n", envvar);
        abort();
    } 
    printf("do_bind %s \n", config );

    int rc = zmq_bind (socket, config );   
    assert (rc == 0); 
}

void do_init( zmq_msg_t* zmsg )
{
    int rc = zmq_msg_init( zmsg ); 
    assert (rc == 0); 
}

void do_copy( zmq_msg_t* dest, zmq_msg_t* src )
{
    int rc = zmq_msg_copy( dest, src );
    if( rc == -1 ){
        int err = zmq_errno();
        printf ("Error occurred during zmq_msg_copy : %s\n", zmq_strerror(err));
        abort (); 
    }
}

void do_receive(zmq_msg_t* zmsg, void* socket)
{
    printf ("do_receive zmq_msg_recv waiting... \n");
    int rc = zmq_msg_recv (zmsg, socket, 0);   
    assert (rc != -1);
    printf("do_receive got bytes:%d \n", rc ); 
}

void do_send( zmq_msg_t* zmsg, void* socket)
{
    size_t size = zmq_msg_size(zmsg);
    printf ("do_send... sending msg of  %zu bytes \n", size );
  
    int rc = zmq_msg_send (zmsg, socket, 0); 

    if (rc == -1) {
        int err = zmq_errno();
        printf ("Error occurred during zmq_msg_send : %s\n", zmq_strerror(err));
        abort (); 
    }   
    printf ("do_send... queued  %d bytes \n", rc );
}

int main (void)
{
    void* context = zmq_ctx_new ();
    void* responder = zmq_socket (context, ZMQ_REP); 
    do_bind( responder, "ECHO_SERVER_CONFIG" );

    while (1) {

        zmq_msg_t zrec;
        zmq_msg_t zsend ; 

        do_init( &zrec );
        do_init( &zsend );

        do_receive( &zrec, responder );
        do_copy( &zsend, &zrec );

        sleep(1);        

        do_send( &zsend, responder );

        zmq_msg_close( &zrec );
        zmq_msg_close( &zsend );
   }

   zmq_close( responder );
   zmq_ctx_destroy( context );

   return 0;
}
