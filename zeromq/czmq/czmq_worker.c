#include "czmq.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>


int main (int argc, char *argv [])
{
    char* backend = getenv("BACKEND");
    zclock_log("I: %s starting ", argv[0]);
    zclock_log("I: REP responder connecting to backend [%s]",backend);

    zctx_t* ctx = zctx_new ();
    void* responder = zsocket_new (ctx, ZMQ_REP);
    zsocket_connect (responder, backend );

    while (true) {

        char* req = zstr_recv( responder );
        if (!req) break;              //  Interrupted
        zclock_log("I: received request: %s", req);
        free (req);

        sleep (1);

        char* rep = "hello czmq";
        zclock_log("I: sending reply: %s", rep);
        zstr_send (responder, rep); 
    }  
 
    zclock_log("I: %s exiting ", argv[0]);
    zctx_destroy (&ctx);
    return 0;

}
