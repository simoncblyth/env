


#include "npyworker.hh"
#include <boost/thread.hpp>
#include <boost/bind.hpp>
#include <thread>
#include "stdio.h"


//
// boost::asio arranges that the async tasks 
// like waiting for network messages
// will get done within threads that called io_service.run() 
// ie here only within m_netThread
//
// note that m_ioService is instanciated in mainThread
// but the run() is called in the netThread 
//
//

class App { 
  public:
      App(const char* backend) 
          :
          //m_ioWork(m_ioService),
          m_npyWorker(m_ioService,m_ctx, backend)
      {
          m_netThread = new boost::thread(boost::bind(&boost::asio::io_service::run, &m_ioService));
      }

  private:
      boost::asio::io_service       m_ioService;
      //boost::asio::io_service::work m_ioWork;
      boost::asio::zmq::context     m_ctx;
      boost::thread*                m_netThread ; 
      npyworker                     m_npyWorker ; 

};




int main(int argc, char* argv[])
{
    // worker instanciated together with the app 
    // but the work is done inside netThread
    // so this main thread does not block 

    App app("tcp://127.0.0.1:5002") ; 


    std::this_thread::sleep_for(std::chrono::seconds(20));

    return 0;
}




