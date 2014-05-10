#include "czmq.h"

int main (int argc, char *argv [])
{
    if (argc < 3) {
        printf ("INFO: syntax: %s <client-REQ-endpoint> <message-string> \n", argv [0]);
        return 0;
    }

    zctx_t* ctx = zctx_new ();
    void* client = zsocket_new (ctx, ZMQ_REQ);
    zsocket_connect (client, argv[1]);

    printf ("INFO: %s client connecting to:[%s] message string [%s] \n", argv[0],argv[1],argv[2]);
    while (true) {
        zstr_send (client, argv[2]);
        char *reply = zstr_recv (client);
        if (!reply)
            break;              //  Interrupted
        printf ("Client got reply: %s\n", reply);
        free (reply);
        sleep (1);
    }   
    zctx_destroy (&ctx);
    return 0;

}
