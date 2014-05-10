
/*
*/


#include "zmq.h"
#include "czmq.h"

int main (int argc, char *argv [])
{
    if (argc < 3) {
        printf ("INFO: syntax: %s <frontend-XREP-endpoint> <backend-XREQ-endpoint> \n", argv [0]);
        printf ("INFO frontend speaks to clients, collecting requests from them \n");
        printf ("INFO backend speaks to services \n");
        return 0;
    }
    zctx_t* ctx = zctx_new ();

    void* frontend = zsocket_new (ctx, ZMQ_XREP);  // XREP
    void* backend  = zsocket_new (ctx, ZMQ_XREQ);  // XREQ
    void* capture = NULL ; 

    zsocket_bind(frontend, argv [1]);
    zsocket_bind(backend,  argv [2]);

    printf ("INFO: %s service starting frontend(XREP):%s backend(XREQ):%s \n", argv[0],argv[1],argv[2]);

    zmq_proxy( frontend, backend, capture );

    zctx_destroy (&ctx);
    return 0;
}
