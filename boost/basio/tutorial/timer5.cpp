/**

   https://think-async.com/Asio/boost_asio_1_18_0/doc/html/boost_asio/tutorial/tuttimer5.html

    env-;basio-;basio-example-make timer5.cpp


This tutorial demonstrates the use of the strand class template to synchronise callback handlers in a multithreaded program.

The previous four tutorials avoided the issue of handler synchronisation by
calling the io_context::run() function from one thread only. As you already
know, the asio library provides a guarantee that callback handlers will only be
called from threads that are currently calling io_context::run(). Consequently,
calling io_context::run() from only one thread ensures that callback handlers
cannot run concurrently.





**/

#include <iostream>
#include <boost/asio.hpp>
#include <boost/thread/thread.hpp>
#include <boost/bind.hpp>


struct printer
{
  boost::asio::strand<boost::asio::io_context::executor_type> strand_;
  boost::asio::steady_timer timer1_;
  boost::asio::steady_timer timer2_;
  int count_;

  printer(boost::asio::io_context& io)
    : strand_(boost::asio::make_strand(io)),
      timer1_(io, boost::asio::chrono::seconds(1)),
      timer2_(io, boost::asio::chrono::seconds(1)),
      count_(0)
  {
    timer1_.async_wait(boost::asio::bind_executor(strand_,boost::bind(&printer::print1, this)));
    timer2_.async_wait(boost::asio::bind_executor(strand_,boost::bind(&printer::print2, this)));

/**
The boost::asio::bind_executor() function returns a new handler that
automatically dispatches its contained handler through the strand object. By
binding the handlers to the same strand, we are ensuring that they cannot
execute concurrently.

In a multithreaded program, the handlers for asynchronous operations should be
synchronised if they access shared resources. In this tutorial, the shared
resources used by the handlers (print1 and print2) are std::cout and the count_
data member.


**/

  }

  ~printer()
  {
    std::cout << "Final count is " << count_ << std::endl;
  }

  void print1()
  {
    if (count_ < 10)
    {
      std::cout << "Timer 1: " << count_ << std::endl;
      ++count_;

      timer1_.expires_at(timer1_.expiry() + boost::asio::chrono::seconds(1));
      timer1_.async_wait(boost::asio::bind_executor(strand_,boost::bind(&printer::print1, this)));
    }
  }

  void print2()
  {
    if (count_ < 10)
    {
      std::cout << "Timer 2: " << count_ << std::endl;
      ++count_;

      timer2_.expires_at(timer2_.expiry() + boost::asio::chrono::seconds(1));
      timer2_.async_wait(boost::asio::bind_executor(strand_,boost::bind(&printer::print2, this)));
    }
  }

};

int main()
{
  boost::asio::io_context io;
  printer p(io);

  boost::thread t(boost::bind(&boost::asio::io_context::run, &io));
  io.run();
  t.join();


/**
The main function now causes io_context::run() to be called from two threads:
the main thread and one additional thread. This is accomplished using an
boost::thread object.

Just as it would with a call from a single thread, concurrent calls to
io_context::run() will continue to execute while there is "work" left to do.
The background thread will not exit until all asynchronous operations have
completed.

**/

  return 0;
}



