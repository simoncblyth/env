# === func-gen- : network/asiozmqtest/asiozmqtest fgp network/asiozmqtest/asiozmqtest.bash fgn asiozmqtest fgh network/asiozmqtest
asiozmqtest-src(){      echo network/asiozmqtest/asiozmqtest.bash ; }
asiozmqtest-source(){   echo ${BASH_SOURCE:-$(env-home)/$(asiozmqtest-src)} ; }
asiozmqtest-vi(){       vi $(asiozmqtest-source) ; }
asiozmqtest-env(){      elocal- ; }
asiozmqtest-usage(){ cat << EOU

ASIO ZMQ Test
===============

Attempting to create C++/boost::asio/zmq/asiozmq 
equivalent of python/pyzmq based NPYResponder

* ~/env/geant4/geometry/collada/g4daeview/daeresponder.py
* ~/env/zeromq/pyzmq/npyresponder.py 
* ~/env/zeromq/pyzmq/npycontext.py 

Just need the worker, the broker and client can stay as is.

* broker (C based zmq server)
* client (numerous pyzmq test clients or C++ G4DAEChroma) 


See also
----------

* asiozmq-
* azmq-


Thoughts
---------

* boost::asio usage inside a netThread probably means 
  can avoid the polling of NPYResponder : 
  just setup async handler to deal with activity on the socket 

* hmm, but npyworker needs to run the propagation
  and then send back the reply  

* both asio-zmq and azmq need C++11 and are header only, 
  that might be a problem

  * using asis forces c++11 switch

  * maybe the C++11 can be isolated by moving into a library 
    and providing a plain C interface for what is needed ?


Testing
---------

broker::

    delta:~ blyth$ zmq-broker
    15-04-01 13:02:27 I: /usr/local/env/bin/zmq_broker start
    15-04-01 13:02:27 I: bind frontend ROUTER:[tcp://*:5001]
    15-04-01 13:02:27 I: bind backend DEALER:[tcp://*:5002]
    15-04-01 13:02:27 I: enter proxy loop


client::

    delta:~ blyth$ zmq-client
    15-04-01 13:02:48 I: /usr/local/env/bin/zmq_client starting
    15-04-01 13:02:48 I: connect REQ socket to frontend [tcp://127.0.0.1:5001]
    15-04-01 13:02:48 send req: REQ HELLO ZMQ
    15-04-01 13:02:49 recv rep: World
    15-04-01 13:02:50 send req: REQ HELLO ZMQ
    ...

worker::

    delta:~ blyth$ asiozmqtest-run
    asiozmqtest backend tcp://127.0.0.1:5002
    npyworker received request: REQ HELLO ZMQ
    npyworker received request: REQ HELLO ZMQ
    npyworker received request: REQ HELLO ZMQ


NPY Client
------------

Succeeds to send to npyworker::

    delta:~ blyth$ npysend.sh --tag 1
    INFO:env.zeromq.pyzmq.npysend:tmplname DAE_CERENKOV_PATH_TEMPLATE tmpl /usr/local/env/cerenkov/%s.npy tag 1 
    INFO:env.zeromq.pyzmq.npysend:load /usr/local/env/cerenkov/1.npy (7836, 6, 4) 
    INFO:env.zeromq.pyzmq.npysend:connect to endpoint tcp://127.0.0.1:5001 
    INFO:env.zeromq.pyzmq.npysend:send_npy
    INFO:env.zeromq.pyzmq.npycontext:send_npy sending 2 bufs copy False 
    INFO:env.zeromq.pyzmq.npycontext:recy_npy got 1 frames: 0 NPY, 0 json metadata, 1 other 
    WARNING:env.zeromq.pyzmq.npycontext:no NPY serialization found in any of the multipart frames 
    INFO:env.zeromq.pyzmq.npysend:response (0,)
    NP([], dtype=float64) 
    INFO:env.zeromq.pyzmq.npysend:tmplname DAE_TEST_PATH_TEMPLATE tmpl /usr/local/env/test/%s.npy tag 1 
    INFO:env.zeromq.pyzmq.npysend:save /usr/local/env/test/1.npy (0,) 
    delta:~ blyth$ 



DAEResponder
--------------

::

     07 from glumpy.window import event
     ..
     10 from env.zeromq.pyzmq.npyresponder import NPYResponder
     ..
     13 class DAEResponder(event.EventDispatcher, NPYResponder):
     ..
     52     def __init__(self, config, scene):
     53         class Cfg(object):
     54             mode = 'connect' # worker, as opposed to 'bind' for server
     55             endpoint = config.args.zmqendpoint
     56             timeout = 100  # millisecond
     57             sleep = 0.5
     58             handler = 'on_external_npy'
     59         pass
     60         cfg = Cfg()
     61         NPYResponder.__init__(self, cfg )


NPYResponder
------------

::

     21 class NPYResponder(object):
     22     """ 
     23     Subclasses need to impleent a reply method
     24     that accepts and returns an obj of the transported class 
     25     """
     26     def __init__(self, config):
     27         context = NPYContext()
     28         socket = context.socket(zmq.REP)
     ..
     33         elif config.mode == 'connect':
     34             log.debug("connect to endpoint [%s] (worker-like) " % config.endpoint )
     35             socket.connect( config.endpoint )
     ..
     42         config.flags = zmq.POLLIN
     ..
     51     def poll(self):
     52         """
     53         https://github.com/zeromq/pyzmq/issues/348
     54         https://gist.github.com/minrk/5258909
     55         """
     56         events = None
     57         try:
     58             events = self.socket.poll(timeout=self.config.timeout, flags=self.config.flags )
     59         except zmq.ZMQError as e:
     60             if e.errno == errno.EINTR:
     61                 log.debug("got zmq.ZMQError : Interrupted System Call, return to poll again sometime, resizing terminal windowtriggers this")
     62                 return
     63             else:
     64                 raise
     65             pass
     66         if events:
     67             request = self.socket.recv_npy(copy=False)
     68             response = self.reply(request)
     69 
     70             self.socket.send_npy(response)


NPYSocket
-----------

::

    021 class NPYSocket(zmq.Socket):
    ...
     31     def send_npy(self, a, flags=0, copy=False, track=False, ipython=False):
     32         """
     33         NPY serialize to a buffer and then send
     45         """
    ...
     71         log.info("send_npy sending %s bufs copy %s " % (len(bufs),copy))
     72         return self.send_multipart( bufs, flags=flags, copy=copy, track=track)
    ...
    ...
    075     def recv_npy(self, flags=0, copy=False, track=False, meta_encoding="ascii", ipython=False):
    ...
    110         else:
    111             frames = self.recv_multipart(flags=flags,copy=False, track=track)
    112             bufs = map(lambda frame:frame.buffer, frames)            # memoryview object 
    113         pass
    ...
    ...         assembles NP array and metadata from the frames 
    ...
    157         aa.meta  = meta
    158         aa.other = other
    159         return aa
    ...
    ...
    161 class NPYContext(zmq.Context):
    162     _socket_class = NPYSocket
    163 



EOU
}
asiozmqtest-sdir(){ echo $(env-home)/network/asiozmqtest ; }
asiozmqtest-bdir(){ echo $(local-base)/env/network/asiozmqtest ; }

asiozmqtest-cd(){   cd $(asiozmqtest-sdir); }
asiozmqtest-scd(){  cd $(asiozmqtest-sdir); }
asiozmqtest-bcd(){  cd $(asiozmqtest-bdir); }

asiozmqtest-bin(){  echo $(asiozmqtest-bdir)/NPYAsioZMQTest ; }

asiozmqtest-wipe(){
   local bdir=$(asiozmqtest-bdir)  ;
   rm -rf $bdir
}
asiozmqtest-cmake(){
   local iwd=$PWD
   local sdir=$(asiozmqtest-sdir) ;
   local bdir=$(asiozmqtest-bdir)  ;
   mkdir -p $bdir
   asiozmqtest-bcd
   cmake $sdir 
   cd $iwd
}
asiozmqtest-make(){
   local iwd=$PWD
   asiozmqtest-bcd
   make $*
   cd $iwd
}
asiozmqtest--(){
   asiozmqtest-wipe
   asiozmqtest-cmake
   asiozmqtest-make
}

asiozmqtest-run(){
   local bin=$(asiozmqtest-bin)
   $bin $*
}

