/**

   https://think-async.com/Asio/boost_asio_1_18_0/doc/html/boost_asio/tutorial/tuttimer3.html

    env-;basio-;basio-example-make timer3.cpp


The steady_timer::async_wait() function expects a handler function (or function
object) with the signature void(const boost::system::error_code&). Binding the
additional parameters converts your print function into a function object that
matches the signature correctly.

   https://www.boost.org/doc/libs/1_74_0/libs/bind/doc/html/bind.html

**/

#include <iostream>
#include <boost/asio.hpp>
#include <boost/bind.hpp>



void print(const boost::system::error_code& /*e*/, boost::asio::steady_timer* t, int* count)
{
  std::cout << "Hello, world!" << std::endl;
  if (*count < 5)
  {
    std::cout << *count << std::endl;
    ++(*count);

    t->expires_at(t->expiry() + boost::asio::chrono::seconds(1));
    t->async_wait(boost::bind(print,boost::asio::placeholders::error, t, count));
  }
}



int main(int argc, char** argv)
{
    boost::asio::io_context io;

    int count = 0;
    boost::asio::steady_timer t(io, boost::asio::chrono::seconds(1));
    t.async_wait(boost::bind(print,boost::asio::placeholders::error, &t, &count));

    io.run();

    std::cout << "Final count is " << count << std::endl;

    return 0 ; 
}



