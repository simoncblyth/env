/*
Based on zguide examples: 

* rrbroker.c
* msgqueue.c
* espresso.c

* http://czmq.zeromq.org/manual:zclock

Apart from zmq_proxy call this is implemented in CZMQ lingo (higher level C dialect).

*/

#include "czmq.h"

//  The listener receives all messages flowing through the proxy, on its
//  pipe. In CZMQ, the pipe is a pair of ZMQ_PAIR sockets that connect
//  attached child threads. 

static void
listener_thread (void *args, zctx_t *ctx, void *pipe)
{
    //  Print everything that arrives on pipe
    while (true) {
        zframe_t *frame = zframe_recv (pipe);
        if (!frame)
            break;              //  Interrupted

        size_t size = zframe_size ( frame );
        zclock_log ("I: listener sees frame of size %zu ", size ); 

        //zframe_print (frame, NULL);
        zframe_destroy (&frame);
    }   
}

/**

* https://zguide.zeromq.org/docs/chapter2/#ZeroMQ-s-Built-In-Proxy-Function

The request-reply broker binds to two endpoints, one for clients to connect to
(the frontend socket) and one for workers to connect to (the backend). 

FRONTEND
    clients make REQ

BACKEND 
     workers make REP 



* DEALER and ROUTER let us extend REQ-REP across an intermediary, that is, our little broker




**/

int main (int argc, char *argv [])
{
    char* frontend_ = getenv("FRONTEND");
    char* backend_ = getenv("BACKEND");

    zclock_log("I: %s starting ", argv[0] );
    zclock_log("I: binding frontend ROUTER:[%s]", frontend_ );
    zclock_log("I: binding backend DEALER:[%s]", backend_ );

    int rc ; 
    zctx_t* ctx = zctx_new ();
 
    void* frontend = zsocket_new(ctx, ZMQ_ROUTER); 
    zsocket_bind(frontend, frontend_);

    void* backend  = zsocket_new(ctx, ZMQ_DEALER);  
    zsocket_bind(backend,  backend_);

    void *listener = zthread_fork (ctx, listener_thread, NULL);

    zmq_proxy( frontend, backend, listener );

    zclock_log("I: %s interrupted : terminating ", argv[0] );
    zctx_destroy (&ctx);

    return 0;

}
