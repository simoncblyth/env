// following /usr/local/env/network/asiozmq/example/rrworker.cpp

#include "npyworker.hh"
#include "numpy.hpp"

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
    //
    // connnect responder socket to backend endpoint (in the broker process), 
    // tee up async_read_message to invoke handle_req
    //
    printf("npyworker::npyworker connecting to backend %s \n", backend);
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

    //m_buffer.clear();
    //m_buffer.push_back(boost::asio::zmq::frame(json));



    // just echoing
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
        if(peek == '{' )
        {
            printf(" JSON \n["); 
            fwrite((char*)frame.data(), sizeof(char), frame.size(), stdout);  // not null terminated
            printf("]\n");

            std::cout << std::to_string(frame) << "\n";
        } 
        else if(peek == '\x93')
        {
            printf(" NPY \n"); 
            dump_npy((char*)frame.data(), frame.size()); 
        }
        else
        {
            printf(" OTHER \n"); 
        }
    }
}

void npyworker::dump_npy( char* bytes, size_t size )
{
    // interpreting (bytes, size)  as serialized NPY array
    std::vector<int>  shape ;
    std::vector<float> data ;
    aoba::BufferLoadArrayFromNumpy<float>(bytes, size, shape, data );

    printf("npyworker::dump_npy data size %lu shape of %lu dimensions : ", data.size(), shape.size());
    int itemsize = 1 ;
    int fullsize = 1 ; 
    for(size_t d=0 ; d<shape.size(); ++d)
    {
       printf("%d ", shape[d]);
       if(d > 0) itemsize *= shape[d] ; 
       fullsize *= shape[d] ; 
    }
    int nitems = shape[0] ; 
    printf("\n itemsize %d fullsize %d nitems %d \n", itemsize, fullsize, nitems);
    assert(fullsize == data.size());


    for(size_t f=0 ; f<data.size(); ++f)
    {
         if(f < itemsize*3 || f >= (nitems - 3)*itemsize )
         {
             if(f % itemsize == 0) printf("%lu\n", f/itemsize);
             printf("%15.4f ", data[f]);
             if((f + 1) % itemsize == 0) printf("\n\n");
             else if((f+1) % 4 == 0) printf("\n");
         }
    }

}





