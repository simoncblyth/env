// Based on zguide examples: rrbroker.c, msgqueue.c

#include "zhelpers.h"

int main (int argc, char *argv [])
{
    char* frontend_ = getenv("FRONTEND");
    char* backend_ = getenv("BACKEND");

    s_console("I: %s start", argv[0] );
    s_console("I: bind frontend ROUTER:[%s]", frontend_ );
    s_console("I: bind backend DEALER:[%s]", backend_ );

    int rc ; 
    void* ctx = zmq_ctx_new ();
 
    void* frontend = zmq_socket(ctx, ZMQ_ROUTER); 
    rc = zmq_bind(frontend, frontend_);
    assert( rc == 0 );

    void* backend  = zmq_socket(ctx, ZMQ_DEALER);  
    rc = zmq_bind(backend,  backend_);
    assert( rc == 0 );

    void* capture = NULL ; 

    s_console("I: enter proxy loop");
    zmq_proxy( frontend, backend, capture );

    // never get here
    zmq_close( frontend );
    zmq_close( backend );
    zmq_ctx_destroy (ctx);
    return 0;
}
