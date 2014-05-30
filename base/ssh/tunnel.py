#!/usr/bin/env python
"""
Specialization of zmq/ssh/tunnel.py assuming OSX/Linux passwordless SSH connections 

* http://zeromq.github.io/pyzmq/ssh.html

"""
import logging, os, socket
log = logging.getLogger(__name__)


from available_port import available_port


def open_tunnel_cmd( addr, server, timeout=60):
    """
    :param addr:
    :param server: 
    """
    lport = available_port()
    transport, addr = addr.split('://')
    ip,rport = addr.split(':')
    rport = int(rport)
    tunnel = openssh_tunnel(lport, rport, server, remoteip=ip, timeout=timeout)
    return 'tcp://127.0.0.1:%i' % lport, tunnel


def openssh_forward(lport, rport, server, remoteip='127.0.0.1'):
    fwd = "127.0.0.1:%(lport)i:%(remoteip)s:%(rport)i" % locals()
    return fwd


def openssh_tunnel(lport, rport, server, remoteip='127.0.0.1', timeout=60, ssh_port=22 ):
    """Create an ssh tunnel using command-line ssh that connects port lport
    on this machine to localhost:rport on server.  The tunnel
    will automatically close when not in use, remaining open
    for a minimum of timeout seconds for an initial connection.
    
    This creates a tunnel redirecting `localhost:lport` to `remoteip:rport`,
    as seen from `server`.
    
    
    Parameters
    ----------
    
    lport : int
        local port for connecting to the tunnel from this machine.
    rport : int
        port on the remote machine to connect to.
    server : str
        The ssh server to connect to. Use ssh config for fine control.
    remoteip : str [Default: 127.0.0.1]
        The remote ip, specifying the destination of the tunnel.
        Default is localhost, which means that the tunnel would redirect
        localhost:lport on this machine to localhost:rport on the *server*.

    timeout : int [default: 60]
        The time (in seconds) after which no activity will result in the tunnel
        closing.  This prevents orphaned tunnels from running forever.

    ssh_port : int

    """
    cmd = "sleep %(timeout)s" % locals()
    #cmd = "-N"  # no command
    cmd = "ssh -f -p %(ssh_port)i -L 127.0.0.1:%(lport)i:%(remoteip)s:%(rport)i %(server)s %(cmd)s" % locals()
    return cmd

def export( **kwa ):
    return "\n".join(["export %s=\"%s\"" % (k, v) for k,v in kwa.items()])


def open_tunnel( url_remote, ssh_server, do=False ):
    url_local, tunnel_cmd = open_tunnel_cmd( url_remote, ssh_server ) 

    if do:
        log.info("opening SSH tunnel in background with command : %s " % cmd )
        rc = os.system( cmd )
        assert rc == 0, "open_tunnel RC %s running cmd %s " % (rc, cmd)
        log.info("open_tunnel completed, url_local : %s " % url_local )
      
    return dict(url=url_local, fwd=tunnel_cmd) 

def main():
    logging.basicConfig(level=logging.INFO)
    import sys
    url_remote, ssh_server = sys.argv[1:]
    d = open_tunnel( url_remote, ssh_server )
    print export(**d)

if __name__ == '__main__':
    main()
    



