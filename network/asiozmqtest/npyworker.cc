// following /usr/local/env/network/asiozmq/example/rrworker.cpp

#include "npyworker.hh"

#include <chrono>
#include <functional>
#include <iostream>
#include <iterator>
#include <thread>

npyworker::npyworker(
                     boost::asio::io_service& ios, 
                     boost::asio::zmq::context& ctx,
                     const char* backend
                    )
                    : 
                     m_responder(ios, ctx, ZMQ_REP), 
                     m_buffer()
{
    m_responder.connect(backend);
    m_responder.async_read_message(
                     std::back_inserter(m_buffer),
                     std::bind(&npyworker::handle_req, this, std::placeholders::_1)
                 );
}

void npyworker::handle_req(boost::system::error_code const& ec)
{
    dump(); 
    std::this_thread::sleep_for(std::chrono::seconds(1));

    m_buffer.clear();
    m_buffer.push_back(boost::asio::zmq::frame("{}"));
    m_responder.write_message(std::begin(m_buffer), std::end(m_buffer));

    m_buffer.clear();
    m_responder.async_read_message(
                     std::back_inserter(m_buffer),
                     std::bind(&npyworker::handle_req, this, std::placeholders::_1)
                 );
}


void npyworker::dump()
{
    for(unsigned int i=0 ; i < m_buffer.size() ; ++i )
    {
        const boost::asio::zmq::frame& frame = m_buffer[i] ; 
        char peek = *(char*)frame.data() ;
        printf("npyworker::dump frame %u/%lu size %8lu peek %c ", i, m_buffer.size(), frame.size(), peek );  
        if(peek == '{')
        {
            printf(" JSON \n"); 
            printf("[%s]\n", (char*)frame.data());
            std::cout << std::to_string(frame) << "\n";
        } 
        else if(peek == '\x93')
        {
            printf(" NPY \n"); 
        }
        else
        {
            printf(" OTHER \n"); 
        }
    }
}
