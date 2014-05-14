/*
Based on zguide examples: rrbroker.c, msgqueue.c

Apart from the zmq_proxy call this is implemented in CZMQ lingo (higher level C dialect).


*/

#include "czmq.h"

int main (int argc, char *argv [])
{
    char* frontend_ = getenv("FRONTEND");
    char* backend_ = getenv("BACKEND");

    printf("INFO: %s starting \n", argv[0] );
    printf("INFO: binding frontend ROUTER:[%s]\n", frontend_ );
    printf("INFO: binding backend DEALER:[%s]\n", backend_ );

    int rc ; 
    zctx_t* ctx = zctx_new ();
 
    void* frontend = zsocket_new(ctx, ZMQ_ROUTER); 
    zsocket_bind(frontend, frontend_);

    void* backend  = zsocket_new(ctx, ZMQ_DEALER);  
    zsocket_bind(backend,  backend_);

    void* capture = NULL ; 

    zmq_proxy( frontend, backend, capture );

    zctx_destroy (&ctx);
    return 0;
}
