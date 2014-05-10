
/*
   Started from zguide example, 
       flserver1.c "Freelance server - Model 1"

::

    import sys
    import zmq

    context = zmq.Context()

    s1 = context.socket(zmq.ROUTER)
    s2 = context.socket(zmq.DEALER)
    s1.bind(sys.argv[1])
    s2.bind(sys.argv[2])
    zmq.device(zmq.QUEUE, s1, s2)



    http://api.zeromq.org/2-1:zmq-device

    ZMQ_QUEUE creates a shared queue that collects requests from a set of clients,
    and distributes these fairly among a set of services. Requests are fair-queued
    from frontend connections and load-balanced between backend connections.
    Replies automatically return to the client that made the original request.

    This device is part of the request-reply pattern. The frontend speaks to
    clients and the backend speaks to services. You should use ZMQ_QUEUE with a
    ZMQ_XREP socket for the frontend and a ZMQ_XREQ socket for the backend. Other
    combinations are not documented.

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

    void* frontend = zsocket_new (ctx, ZMQ_ROUTER);  // XREP
    void* backend  = zsocket_new (ctx, ZMQ_DEALER);  // XREQ

    zsocket_connect (frontend, argv [1]);
    zsocket_connect (backend,  argv [2]);

    printf ("INFO: %s service starting frontend(XREP):%s backend(XREQ):%s \n", argv[0],argv[1],argv[2]);

    zmq_device( ZMQ_QUEUE,  frontend, backend );

    zctx_destroy (&ctx);
    return 0;
}
