/**
https://dens.website/tutorials/cpp-asio/simplest-server

Build and run the server::

     boost-;gcc udp_echo_server.cpp -I$(boost-prefix)/include -lstdc++ -o /tmp/udp_echo_server && /tmp/udp_echo_server

Send it a UDP message and get it echoed back::

    epsilon:env blyth$ UDP_PORT=15001 udp.py hello world from udp.py 
    sending [hello world from udp.py] to host:port 127.0.0.1:15001 
    received b'hello world from udp.py' 
    epsilon:env blyth$ 

**/

#include <iostream>
#include <boost/asio.hpp>

int main()
{
    std::uint16_t port = 15001;

    boost::asio::io_context io_context;
    boost::asio::ip::udp::endpoint receiver(boost::asio::ip::udp::v4(), port);
    boost::asio::ip::udp::socket socket(io_context, receiver);

    for(;;)
    {
        char buffer[65536];
        boost::asio::ip::udp::endpoint sender;
        std::size_t bytes_transferred = socket.receive_from(boost::asio::buffer(buffer), sender);
        std::cout << " bytes_transferred " << bytes_transferred << std::endl  ; 
        socket.send_to(boost::asio::buffer(buffer, bytes_transferred), sender);
    }

    return 0;
}



