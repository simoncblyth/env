/**

https://think-async.com/Asio/boost_asio_1_18_0/doc/html/boost_asio/tutorial/tutdaytime1.html

    env-;basio-;basio-example-make daytime1_client.cpp

**/

//
// client.cpp
// ~~~~~~~~~~
//
// Copyright (c) 2003-2020 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#include <iostream>
#include <boost/array.hpp>
#include <boost/asio.hpp>

using boost::asio::ip::tcp;

int main(int argc, char* argv[])
{
  try
  {
    if (argc != 2)
    {
      std::cerr << "Usage: client <host>" << std::endl;
      return 1;
    }

    boost::asio::io_context io_context;

    tcp::resolver resolver(io_context);
    tcp::resolver::results_type endpoints = resolver.resolve(argv[1], "daytime");
/**
A resolver takes a host name and service name and turns them into a list of
endpoints. We perform a resolve call using the name of the server, specified in
argv[1], and the name of the service, in this case "daytime".

The list of endpoints is returned using an object of type
ip::tcp::resolver::results_type. This object is a range, with begin() and end()
member functions that may be used for iterating over the results.

**/
    tcp::socket socket(io_context);
    boost::asio::connect(socket, endpoints);
/**
Now we create and connect the socket. The list of endpoints obtained above may
contain both IPv4 and IPv6 endpoints, so we need to try each of them until we
find one that works. This keeps the client program independent of a specific IP
version. The boost::asio::connect() function does this for us automatically.
**/
    for (;;)
    {
      boost::array<char, 128> buf;
      boost::system::error_code error;

      size_t len = socket.read_some(boost::asio::buffer(buf), error);
/**
We use a boost::array to hold the received data. The boost::asio::buffer()
function automatically determines the size of the array to help prevent buffer
overruns. Instead of a boost::array, we could have used a char [] or
std::vector.
**/

      if (error == boost::asio::error::eof)
        break; // Connection closed cleanly by peer.
      else if (error)
        throw boost::system::system_error(error); // Some other error.

      std::cout.write(buf.data(), len);
    }
  }
  catch (std::exception& e)
  {
    std::cerr << e.what() << std::endl;
  }

  return 0;
}


