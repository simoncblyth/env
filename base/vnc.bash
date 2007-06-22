vnc-tunnel(){

  ## ssh username@hostname -L 5901/127.0.0.1/5901
  ssh username@hostname -L 5901/127.0.0.1/5901
  
  #
  #
  #
  #
  #
  #
  #  [bind_address:]port:host:hostport]
  #
  #  -L [bind_address:]port:host:hostport
  #           Specifies that the given port on the local (client) host is to be
  #           forwarded to the given host and port on the remote side.  This
  #           works by allocating a socket to listen to port on the local side,
  #           optionally bound to the specified bind_address.  Whenever a con-
  #           nection is made to this port, the connection is forwarded over
  #           the secure channel, and a connection is made to host port
  #           hostport from the remote machine.  Port forwardings can also be
  #           specified in the configuration file.  IPv6 addresses can be spec-
  #           ified with an alternative syntax:
  #           [bind_address/]port/host/hostport or by enclosing the address in
  #           square brackets.  Only the superuser can forward privileged
  #           ports.  By default, the local port is bound in accordance with
  #           the GatewayPorts setting.  However, an explicit bind_address may
  #           be used to bind the connection to a specific address.  The
  #           bind_address of ``localhost'' indicates that the listening port
  #           be bound for local use only, while an empty address or `*' indi-
  #           cates that the port should be available from all interfaces.
  #
  #
  #
  #
  # For example, if you issue the command
  #
  # 
  # ssh2 -L 1234:localhost:23 username@host
  #  all traffic coming to port 1234 on the client will be forwarded to port 23 on the server (host). 
  # Note that localhost will be resolved by the sshdserver after the connection is established. 
  # In this case localhost therefore refers to the server (host) itself.
  #
  #   man ssh_config
  #  
  #     LocalForward
  #           Specifies that a TCP port on the local machine be forwarded over
  #           the secure channel to the specified host and port from the remote
  #           machine.  The first argument must be [bind_address:]port and the
  #           second argument must be host:hostport.  IPv6 addresses can be
  #           specified by enclosing addresses in square brackets or by using
  #           an alternative syntax: [bind_address/]port and host/hostport.
  #           Multiple forwardings may be specified, and additional forwardings
  #           can be given on the command line.  Only the superuser can forward
  #           privileged ports.  By default, the local port is bound in accor-
  #           dance with the GatewayPorts setting.  However, an explicit
  #           bind_address may be used to bind the connection to a specific
  #           address.  The bind_address of ``localhost'' indicates that the
  #           listening port be bound for local use only, while an empty
  #           address or `*' indicates that the port should be available from
  #           all interfaces.
  #
  #
  #
  #
  # Host=pashar.dyndns.org
  # CheckHostIP=no
  # Compression=yes
  # CompressionLevel=9
  # Port=995
  # Protocol=2
  # LocalForward	4080 127.0.0.1:4080
  # LocalForward	5801 127.0.0.1:5801
  #
  # DynamicForward	2000
  # GatewayPorts=yes
  
  
  
  
  
}