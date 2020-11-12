// boost-;gcc server0.cpp -I$(boost-prefix)/include -lstdc++ -o /tmp/server0  
// https://stackoverflow.com/questions/24149741/client-server-between-c-and-python-programs-using-boost-asio/24379091

#include <iostream>
#include <string>
#include <boost/asio.hpp>

namespace ba = boost::asio ; 
namespace ip = boost::asio::ip  ; 

int main(int argc, char** argv)
{
    const int SERVER_PORT = 50013;
    try 
    {
        ba::io_service io_service;
        ip::tcp::endpoint endpoint(ip::tcp::v4(),SERVER_PORT);
        ip::tcp::acceptor acceptor(io_service, endpoint);
        ip::tcp::socket socket(io_service);

        std::cout << "Server ready" << std::endl;
        {
            acceptor.accept(socket);
            int foo [5] = { 16, 2, 77, 40, 12071 }; 
            ba::write(socket, ba::buffer(foo));
            socket.close();
        }
    }
    catch(std::exception& ex)
    {
        std::cerr << "Exception: " << ex.what() <<std::endl;
    }
    return 0;
}
