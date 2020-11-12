/**
https://stackoverflow.com/questions/3571156/why-doesnt-this-boost-asio-code-work-with-this-python-client

// boost-;gcc server.cpp -I$(boost-prefix)/include -lstdc++ -o /tmp/server

**/

#include <cstdlib>
#include <iostream>
#include <boost/bind.hpp>
#include <boost/asio.hpp>

namespace ba = boost::asio ; 

using ba::ip::udp;
//namespace udp = ba::ip::udp ;

class server
{
   public:
  server(ba::io_service& io_service, short port)
    : io_service_(io_service),
      socket_(io_service, udp::endpoint(udp::v4(), port)),
      socket2_(io_service, udp::endpoint(udp::v4(),0))
  {
    socket_.async_receive_from(
        ba::buffer(data_, max_length), sender_endpoint_,
        boost::bind(&server::handle_receive_from, this,
          ba::placeholders::error,
          ba::placeholders::bytes_transferred));
  }

  void handle_receive_from(const boost::system::error_code& error,
      size_t bytes_recvd)
  {
    if (!error && bytes_recvd > 0)
    {
        // use a different socket... random source port.
        socket2_.async_send_to(
            ba::buffer(data_, bytes_recvd), sender_endpoint_,
            boost::bind(&server::handle_send_to, this,
                        ba::placeholders::error,
                        ba::placeholders::bytes_transferred));
    }
    else
    {
      socket_.async_receive_from(
          ba::buffer(data_, max_length), sender_endpoint_,
          boost::bind(&server::handle_receive_from, this,
            ba::placeholders::error,
            ba::placeholders::bytes_transferred));
    }
  }

  void handle_send_to(const boost::system::error_code& /*error*/,
      size_t /*bytes_sent*/)
  {
    // error_code shows success when checked here.  But wireshark shows
    // an ICMP response with destination unreachable, port unreachable when run on
    // localhost.  Haven't tried it across a network.

    socket_.async_receive_from(
        ba::buffer(data_, max_length), sender_endpoint_,
        boost::bind(&server::handle_receive_from, this,
          ba::placeholders::error,
          ba::placeholders::bytes_transferred));
  }

private:
  ba::io_service& io_service_;
  udp::socket socket_;
  udp::socket socket2_;
  udp::endpoint sender_endpoint_;
  enum { max_length = 1024 };
  char data_[max_length];
};

int main(int argc, char* argv[])
{
  try
  {
    if (argc != 2)
    {
      std::cerr << "Usage: async_udp_echo_server <port>\n";
      return 1;
    }

    ba::io_service io_service;

    using namespace std; // For atoi.
    server s(io_service, atoi(argv[1]));

    io_service.run();
  }
  catch (std::exception& e)
  {
    std::cerr << "Exception: " << e.what() << "\n";
  }

  return 0;
}
