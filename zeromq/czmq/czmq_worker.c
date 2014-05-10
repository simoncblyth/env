#include "czmq.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>


int main (int argc, char *argv [])
{
    char* backend = getenv("BACKEND");
    printf ("INFO: %s connecting to backend [%s]   \n", argv[0],backend);

    zctx_t* ctx = zctx_new ();
    void* responder = zsocket_new (ctx, ZMQ_REP);
    zsocket_connect (responder, backend );

    while (true) {

        char* req = zstr_recv( responder );
        if (!req) break;              //  Interrupted
        printf ("%s received request: %s\n", argv[0], req);
        free (req);

        sleep (1);

        char* rep = "hello czmq";
        printf ("%s sending reply: %s\n", argv[0], rep);
        zstr_send (responder, rep); 
    }   
    zctx_destroy (&ctx);
    return 0;

}
