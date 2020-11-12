/**

   https://think-async.com/Asio/boost_asio_1_18_0/doc/html/boost_asio/tutorial/tuttimer4.html

    env-;basio-;basio-example-make timer4.cpp


The steady_timer::async_wait() function expects a handler function (or function
object) with the signature void(const boost::system::error_code&). Binding the
additional parameters converts your print function into a function object that
matches the signature correctly.

   https://www.boost.org/doc/libs/1_74_0/libs/bind/doc/html/bind.html

**/

#include <iostream>
#include <boost/asio.hpp>
#include <boost/bind.hpp>


class printer
{
public:
    printer(boost::asio::io_context& io)
    : 
    timer_(io, boost::asio::chrono::seconds(1)),
    count_(0)
    {
        timer_.async_wait(boost::bind(&printer::print, this));
    }

 ~printer()
  {
    std::cout << "Final count is " << count_ << std::endl;
  }

  void print()
  {
    if (count_ < 5)
    {
      std::cout << count_ << std::endl;
      ++count_;

      timer_.expires_at(timer_.expiry() + boost::asio::chrono::seconds(1));
      timer_.async_wait(boost::bind(&printer::print, this));
    }
  }

private:
  boost::asio::steady_timer timer_;
  int count_;
};



int main(int argc, char** argv)
{
    boost::asio::io_context io;
    printer p(io);
    io.run();
    return 0 ; 
}



