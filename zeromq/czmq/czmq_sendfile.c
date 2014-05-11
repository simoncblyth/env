/*

Based on:

  /usr/local/env/zeromq/zguide/examples/C/fileio1.c 
  /usr/local/env/zeromq/zguide/examples/C/titanic.c
  http://czmq.zeromq.org/manual:zfile 
  http://czmq.zeromq.org/manual:zmsg 


delta:czmq blyth$ FRONTEND=tcp://203.64.184.126:5001 /usr/local/env/bin/czmq_sendfile /tmp/lastmsg.zmq
I: /usr/local/env/bin/czmq_sendfile starting 
I: /tmp/lastmsg.zmq load file 
czmq_sendfile(8892,0x7fff7ab15310) malloc: *** mach_vm_map(size=288230376151715840) failed (error code=3)
*** error: can't allocate region
*** set a breakpoint in malloc_error_break to debug
Segmentation fault: 11


*/

#include "czmq.h"
#include "assert.h"

#define CHUNK_SIZE  250000

int main (int argc, char *argv [])
{
    printf ("I: %s starting \n", argv[0] );
    printf ("I: %s load file \n", argv[1] );

    assert( zfile_exists (argv[1])) ;
    FILE *file = fopen (argv[1], "r");
    assert (file);

    zmsg_t* req  = zmsg_load (NULL, file);
    assert ( req != NULL );  // msg could not be loaded

    fclose (file);

    char* frontend = getenv("FRONTEND");
    printf ("I: connect REQ socket to frontend [%s]\n", frontend);

    zctx_t *ctx = zctx_new ();
    void* requester = zsocket_new (ctx, ZMQ_REQ);
    zsocket_connect (requester, frontend);

    printf ("I: zmsg_send \n");
    zmsg_send( &req, requester );

    zmsg_t* rep = zmsg_recv( requester ); 

    printf ("I: done\n");

    zctx_destroy (&ctx);

    return 0;
}


