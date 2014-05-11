// a worker that writes messages to file

/*

int zmsg_save (zmsg_t *self, FILE *file);

    Save message to an open file, return 0 if OK, else -1. The message is
    saved as a series of frames, each with length and data. Note that the
    file is NOT guaranteed to be portable between operating systems, not
    versions of CZMQ. The file format is at present undocumented and liable
    to arbitrary change.

*/


#include "czmq.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

int main (int argc, char *argv [])
{
    int rc ; 
    char* backend = getenv("BACKEND");
    printf ("INFO: %s connecting to backend [%s]   \n", argv[0],backend);

    zctx_t* ctx = zctx_new ();
    void* responder = zsocket_new (ctx, ZMQ_REP);
    zsocket_connect (responder, backend );

    while (true) {

        zmsg_t* req = zmsg_recv( responder ); 
        if (!req) break;              //  Interrupted

        FILE *file = fopen (argv[1], "w");
        assert (file);

        rc = zmsg_save( req, file );
        assert( rc == 0);
    
        fclose (file);

        sleep (1);

        zmsg_t* rep = zmsg_dup (req);

        zmsg_send (&rep, responder); 


    }   
    zctx_destroy (&ctx);
    return 0;

}
