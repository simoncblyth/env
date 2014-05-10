#include "czmq.h"

int main (int argc, char *argv [])
{
    char* frontend = getenv("FRONTEND");
    printf ("I: %s starting \n", argv[0] );
    printf ("I: connect REQ socket to frontend [%s]\n", frontend);

    zctx_t* ctx = zctx_new ();
    void* requester = zsocket_new (ctx, ZMQ_REQ);
    zsocket_connect (requester, frontend);

    while (true) {
        char* req = "REQ HELLO CZMQ" ;
        printf ("send req: %s\n", req);
        zstr_send (requester, req );

        sleep (1);

        char *rep = zstr_recv (requester);
        if (!rep) break;              //  Interrupted
        printf ("recv rep: %s\n", rep);
        free (rep);
    }   
    zctx_destroy (&ctx);
    return 0;

}
