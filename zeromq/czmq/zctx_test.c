#include <assert.h>
#include <czmq.h>


int main(void)
{

    zctx_t *ctx = zctx_new ();
    assert (ctx);
    zctx_destroy (&ctx);
    assert (ctx == NULL);

    //  Create a context with many busy sockets, destroy it
    ctx = zctx_new ();
    assert (ctx);
    zctx_set_iothreads (ctx, 1);
    zctx_set_linger (ctx, 5);       //  5 msecs
    void *s1 = zctx__socket_new (ctx, ZMQ_PAIR);
    void *s2 = zctx__socket_new (ctx, ZMQ_XREQ);
    void *s3 = zctx__socket_new (ctx, ZMQ_REQ);
    void *s4 = zctx__socket_new (ctx, ZMQ_REP);
    void *s5 = zctx__socket_new (ctx, ZMQ_PUB);
    void *s6 = zctx__socket_new (ctx, ZMQ_SUB);
    zsocket_connect (s1, "tcp://127.0.0.1:5555");
    zsocket_connect (s2, "tcp://127.0.0.1:5555");
    zsocket_connect (s3, "tcp://127.0.0.1:5555");
    zsocket_connect (s4, "tcp://127.0.0.1:5555");
    zsocket_connect (s5, "tcp://127.0.0.1:5555");
    zsocket_connect (s6, "tcp://127.0.0.1:5555");
    assert (zctx_underlying (ctx)); 
    zctx_destroy (&ctx);

    return 0 ; 
}

