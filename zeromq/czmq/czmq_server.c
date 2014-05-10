#include "czmq.h"

int main (int argc, char *argv [])
{
    if (argc < 2) {
        printf ("INFO: syntax: %s <server-REP-endpoint> \n", argv [0]);
        return 0;
    }
    zctx_t* ctx = zctx_new ();
    void* server = zsocket_new (ctx, ZMQ_REP);
    zsocket_bind (server, argv[1]);

    printf ("INFO: %s server binding to:[%s]   \n", argv[0],argv[1]);

    while (true) {
        char* req = zstr_recv(server);
        if (!req) break;              //  Interrupted
        printf ("Server got request: %s\n", req);

        sleep (1);
        zstr_send (server, req); 
        free (req);
    }   
    zctx_destroy (&ctx);
    return 0;

}
