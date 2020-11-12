/**
https://dens.website/tutorials/cpp-asio/async-tcp-server

Build and run the server::

     boost-;clang++ tcp_async_server.cpp -std=c++14 -I$(boost-prefix)/include -lc++ -o /tmp/tcp_async_server && /tmp/tcp_async_server

Say hello::

    epsilon:bin blyth$ TCP_PORT=15001 tcp.py hello via tcp.py 
    sock.connect to host:port epsilon.local:15001 
    sock.sendall [hello via tcp.py] 
    ^CTraceback (most recent call last):
      File "/Users/blyth/env/bin/tcp.py", line 28, in <module>
        main()
      File "/Users/blyth/env/bin/tcp.py", line 20, in main
        data = sock.recv(4096)
    KeyboardInterrupt


**/

#include <iostream>
#include <experimental/optional>
#include <boost/asio.hpp>

class session : public std::enable_shared_from_this<session>
{
public:

    session(boost::asio::ip::tcp::socket&& socket)
    : socket(std::move(socket))
    {
    }

    void start()
    {
        boost::asio::async_read_until(socket, streambuf, '\n', [self = shared_from_this()] (boost::system::error_code error, std::size_t bytes_transferred)
        {
            std::cout << std::istream(&self->streambuf).rdbuf();
        });
    }

private:

    boost::asio::ip::tcp::socket socket;
    boost::asio::streambuf streambuf;
};

class server
{
public:

    server(boost::asio::io_context& io_context, std::uint16_t port)
    : io_context(io_context)
    , acceptor  (io_context, boost::asio::ip::tcp::endpoint(boost::asio::ip::tcp::v4(), port))
    {
        std::cout << "server::server" << std::endl ; 
    }

    void async_accept()
    {
        std::cout << "server::async_accept" << std::endl ; 
        socket.emplace(io_context);

        acceptor.async_accept(*socket, [&] (boost::system::error_code error)
        {
            std::make_shared<session>(std::move(*socket))->start();
            async_accept();
        });
    }

private:

    boost::asio::io_context& io_context;
    boost::asio::ip::tcp::acceptor acceptor;
    std::experimental::optional<boost::asio::ip::tcp::socket> socket;
};

int main()
{
    boost::asio::io_context io_context;
    server srv(io_context, 15001);
    srv.async_accept();
    io_context.run();
    return 0;
}
