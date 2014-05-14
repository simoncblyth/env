#include "czmq.h"

int main (int argc, char *argv [])
{
    char* frontend = getenv("FRONTEND");
    zclock_log("I: %s starting ", argv[0] );
    zclock_log("I: connect REQ socket to frontend [%s]", frontend);

    zctx_t* ctx = zctx_new ();
    void* requester = zsocket_new (ctx, ZMQ_REQ);
    zsocket_connect (requester, frontend);

    while (true) {
        char* req = "REQ HELLO CZMQ" ;
        zclock_log("I: send req: %s", req);
        zstr_send (requester, req );

        sleep (1);

        char *rep = zstr_recv (requester);
        if (!rep) break;              //  Interrupted
        zclock_log("I: recv rep: %s", rep);
        free (rep);
    }   
    zclock_log("I: %s exiting", argv[0]);
    zctx_destroy (&ctx);
    return 0;

}
