
#include "npyworker.hh"
#include "stdio.h"

int main(int argc, char* argv[])
{
    const char* backend = "tcp://127.0.0.1:5002" ;
    printf("asiozmqtest backend %s\n", backend);

    boost::asio::io_service   ios;
    boost::asio::zmq::context ctx;

    npyworker worker(ios, ctx, backend);

    ios.run();

    return 0;
}




