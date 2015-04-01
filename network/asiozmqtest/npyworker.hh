#ifndef NPYWORKER_H
#define NPYWORKER_H

#include <vector>
#include <boost/asio.hpp>
#include <asio-zmq.hpp>

// following /usr/local/env/network/asiozmq/example/rrworker.cpp

class npyworker {
public:
    npyworker(
             boost::asio::io_service& ios, 
             boost::asio::zmq::context& ctx,
             const char* backend
            );

    void handle_req(boost::system::error_code const& ec);
    void dump();

private:
    boost::asio::zmq::socket             m_responder;
    std::vector<boost::asio::zmq::frame> m_buffer  ;


};

#endif

