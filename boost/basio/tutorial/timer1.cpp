/**

   https://think-async.com/Asio/boost_asio_1_18_0/doc/html/boost_asio/tutorial/tuttimer1.html

    env-;basio-;basio-example-make timer1.cpp


**/

#include <iostream>
#include <boost/asio.hpp>

int main(int argc, char** argv)
{
    boost::asio::io_context io;

    boost::asio::steady_timer t(io, boost::asio::chrono::seconds(5));

    t.wait();

    std::cout << "Hello, world! : " << argv[0] << std::endl;

    return 0 ; 
}



