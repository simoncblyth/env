#!/usr/bin/env python
"""
Extract from zmq/ssh/tunnel.py
"""

import socket

_random_ports = set()

def available_ports(n):
    """Selects and return n random ports that are available."""
    ports = []
    for i in range(n):
        sock = socket.socket()
        sock.bind(('', 0))
        while sock.getsockname()[1] in _random_ports:
            sock.close()
            sock = socket.socket()
            sock.bind(('', 0))
        ports.append(sock)
    for i, sock in enumerate(ports):
        port = sock.getsockname()[1]
        sock.close()
        ports[i] = port
        _random_ports.add(port)
    return ports

def available_port():
    return available_ports(1)[0]

def main():
    print(available_port())

if __name__ == '__main__':
    main()
