#!/usr/bin/env python
"""
Exercising pika and demoing a threading problem 
   http://dayabay.phys.ntu.edu.tw/tracs/env/ticket/320


"""
import os, sys, platform, time, datetime, threading, multiprocessing, Queue, pickle, atexit
from ConfigParser import ConfigParser
import pika

class Envelope(object):
    """
    Message manipulator/converter
    """
    def __init__(self, obj , **opts ):
        if issubclass( obj.__class__ , Envelope ):
            raise Exception("already in Envelope")
        self.obj = obj
        self.opts = opts 
    def _prop(self):
        return pika.BasicProperties( content_type=self.opts.get('content_type','text/plain'), delivery_mode=self.opts.get('delivery_mode', 1)) 
    properties = property(_prop)
    def _body(self):
        return self.obj
    body = property(_body)

 
class T(threading.Thread):
    """
    Thread runner for Worker subclasses
    Using all daemon threads allows the MainThread to exit while the daemons are still alive 
     ... which means should run from ipython to keep it alive.

    Making workers stoppable breaks agnosticism
    """
    def __init__( self,  cls, args, kwargs ):
        q = kwargs.pop('q',None) or Queue.Queue()
        w = cls( q, *args, **kwargs )
        threading.Thread.__init__(self, name=kwargs.get('name',"%s%s" % (self.__class__.__name__, cls.__name__)))  
        self.daemon = False
        self.q = q 
        self.w = w 

    def run(self):
        self.w()

class P(multiprocessing.Process):
    """ 
    Process runner for Worker subclasses
    """
    def __init__( self, cls, args, kwargs):
        q = kwargs.pop('q', None) or  multiprocessing.Queue()
        w = cls( q, *args, **kwargs )
        multiprocessing.Process.__init__(self, name=kwargs.get('name',"%s%s" % (self.__class__.__name__, cls.__name__ )))  
        self.q = q 
        self.w = w 

    def run(self):
        self.w()

class Worker(object):
    """
    Worker conventions :
        * stay thread/process agnostic
        * blocking call in the __call__
        * 1st __init__ argument is the appropriate q instance
        * 2nd __init__ argument is dict like object such as PikaOpts
          which gets updated by kwargs 
    """
    tag = property( lambda self:self.opts.get('variant', self.__class__.__name__ )  ) 
    def __init__(self, q, opts, **kwargs ):
        self.q = q 
        self.opts = opts
        self.opts.update( kwargs )

class Dumper(Worker):          
    """
    Dumps objects appearing in the local q 
    """
    def __call__(self):             
        pika.log.debug("%s starting ", self.tag )
        while True:
            m = self.q.get()  # blocks 
            pika.log.info( "%s %r ", self.tag,  m ) 


class Emitter(Worker):          
    """
    Puts strings onto the argument q 
    every interval seconds ...

    Usage::
        emit = Emitter( q, interval=10 )
        emit()

    """
    interval = property( lambda self:self.opts.get('interval',5) )
    def __call__(self):             
        pika.log.info("%s starting ", self.tag )
        self.countdown = 10 
        while self.countdown > 0:
            self.countdown -= 1 
            now = datetime.datetime.now() 
            msg = "Emitter-%i-%s" % ( self.countdown, now.strftime("%c") )
            self.q.put( msg )  
            time.sleep( self.interval )
        pika.log.info("%s finishing ", self.tag )


class Consumer(Worker):
    """
    Connects to a RabbitMQ server and via a sequence of callbacks 
    wires up an AMQP exchange and queue

    Messages appearing on the remote queue are propagated onto the local queue
    """
    def __call__(self):
        self.count = 0
        self.last_count = None
        self.last_time = None
        self.channel  = None
        self.connection = None

        self.connect()
        self.connection.ioloop.start()

    def connect(self):
        """
        Note the low level way of setting the on_open callback in order to specify one_shot=True 
        otherwise the publisher ends up behaving as a 2nd consumer ... leading to confusion/roundrobining/duplication 

        """ 
        pika.log.debug("%s : connect " , self.tag )
        credentials = pika.PlainCredentials(self.opts.username, self.opts.password )
        parameters = pika.ConnectionParameters(self.opts.host , credentials=credentials)
        connection = pika.SelectConnection( parameters )

        connection.callbacks.add(0, '_on_connection_open', self.on_connected , one_shot=True )
        connection.add_on_close_callback( self.on_closed )
        self.connection = connection


    def cb(self, name):
        """
        Callback variants postfixed with lowercased classname 
        take precedence, eg 
             on_exchange_declared_consumer
             on_exchange_declared_publisher

        """
        names = [ "%s_%s" % ( name, self.__class__.__name__.lower() ), name ]
        for name in names:
            if hasattr(self, name ):
                return getattr(self, name)
        return self.fallback 

    def fallback(self, *args ):
        pika.log.warning("%s: fallback callback invoked %r " % ( self.tag , args ) )

    def on_connected(self, connection):
        pika.log.info("%s: connected : %r " % ( self.tag , connection) )
        self.connection = connection
        self.connection.channel( self.on_channel_open )

    def on_channel_open(self,channel):
        pika.log.info("%s: channel_open : %r " % ( self.tag , channel )  )
        self.channel = channel
        args = dict( type=self.opts.exchange_type, exchange=self.opts.exchange_name, durable=self.opts.durable , auto_delete=self.opts.auto_delete , callback=self.cb('on_exchange_declared') ) 
        pika.log.info("%s: exchange_declare start : %r ", self.tag, args )
        channel.exchange_declare( **args )

    def on_closed(self, connection):
        pika.log.info("%s: closing connection : %r " % ( self.tag , connection) )
        connection.ioloop.stop()

    def close(self):
        connection = self.connection
        connection.close() 
        connection.ioloop.start() # Loop until we're fully closed, will stop on its own

    def on_exchange_declared_consumer(self, frame):
        pika.log.info("%s: exchange_declared_consumer : %r " % ( self.tag , frame )  )
        self.channel.queue_declare( queue=self.opts.queue_name, durable=self.opts.durable , exclusive=self.opts.exclusive , auto_delete=self.opts.auto_delete , callback=self.on_queue_declared )
            
    def on_exchange_declared_publisher(self, frame):
        """ Getting the obj from the queue blocks until available """
        pika.log.info("%s: exchange_declared_publisher : %r " % ( self.tag , frame )  )
        while True:
            obj = self.q.get()  
            evl = Envelope(obj)
            pika.log.info( "%s %r ", self.tag , evl ) 
            self.channel.basic_publish( exchange=self.opts.exchange_name , routing_key=self.opts.routing_key , body=evl.body, properties=evl.properties ) 

    def on_queue_declared(self, frame): 
        pika.log.info("%s: queue_declared : %r " % ( self.tag, frame )  )
        channel = self.channel
        channel.queue_bind( queue=self.opts.queue_name, exchange=self.opts.exchange_name, routing_key=self.opts.routing_key, callback=self.on_queue_bound )

    def on_queue_bound(self, frame):
        pika.log.info("%s: ready : %r " % ( self.tag, frame ) )
        self.start_time = time.time()
        channel = self.channel
        channel.basic_consume( self.on_handle_delivery , queue=self.opts.queue_name , no_ack=self.opts.no_ack )

    def on_handle_delivery(self, channel, method_frame, header_frame, body):
        pika.log.info(" %s : handle_delivery : channel %r  method %r header %r body %r " , self.tag, channel, method_frame, header_frame, body )
        pika.log.info(" %s : content_type %s delivery_tag %s " , self.tag, header_frame.content_type, method_frame.delivery_tag )

        self.q.put( body )

        if not self.opts.no_ack:
            channel.basic_ack(delivery_tag=method_frame.delivery_tag)

        count = self.count 
        count += 1

        now = time.time()
        duration = now - (self.last_time or 0)
        received = count - (self.last_count or 0)
        rate = received / duration
           
        self.last_time = now
        self.last_count = count
        self.count = count          
 
        pika.log.info(" %s : timed_receive: %i Messages Received, %.4f per second", self.tag, self.count, rate)


class Publisher(Consumer):
    """
    Connects to a RabbitMQ server and via a sequence of callbacks 
    wires up an AMQP exchange and queue

    Messages appearing on the local queue are propagated onto the remote queue

    The behavior split between Publisher and Consumer 
    arises from the classname influencing the chain of callbacks in force
    """
    pass
 

class PikaOpts(dict):
    defaults = dict( 
                  host='127.0.0.1', username='guest', password='guest' , vhost='/' , port='5672' ,
                  queue_name='pmq' , exchange_name="pmx" , exchange_type="topic" , routing_key="pmq.#" , 
                  durable=True, exclusive=False, auto_delete=False, no_ack=False  
                  )
    def __getattr__(self, name ):
        return self.get(name, self.defaults.get(name, None)) 

    def __repr__(self):
        d = self.copy()
        d.update( host="***" , password="***", username="***" )
        return repr(d)

    def read(self, cnf='local', path='~/.rmq.cnf' ):
        path = os.path.expanduser(path)
        if os.path.exists( path ): 
            cfp = ConfigParser()
            cfp.read( path )
            if cfp.has_section(cnf):
                for o in cfp.options( cnf ):
                    self[o] = cfp.get( cnf, o )
        return self
    pass


class PikaMQ(object):
    """
    Umbrella class to hold 

    cons
         consumer that propagates messages from remote AMQP queue onto local queue 
    pubr
         publisher that propagates messages placed on local queue to remote AMQP exchange
    dmpr
         dumper that dumps objects arriving on the local queue
    emtr 
         emitter that puts messages onto the local queue every interval seconds


    Fails when Consumer and Publisher are both threads .. 


    """
    def __init__(self, cnf='mon' , fail=True ):
        pika.log.setup(color=True, level=pika.log.INFO )    
        opts = PikaOpts().read(cnf)
        pika.log.info( "PikaOpts %r ", opts )
        atexit.register( PikaMQ.cleanup )

        if fail:
            cons = T( Consumer,  (opts,) , {} ) 
            pubr = T( Publisher, (opts,) , {} )
            dmpr = T( Dumper,    (opts,),  dict(q=cons.q,) )
            emtr = T( Emitter,   (opts,),  dict(q=pubr.q,) )
        else:
            cons = P( Consumer,  (opts,) , {} )   
            pubr = T( Publisher, (opts,) , {} )
            dmpr = T( Dumper,    (opts,),  dict(q=cons.q,) )
            emtr = T( Emitter,   (opts,),  dict(q=pubr.q,) )
 

        self.opts = opts     
   
        self.cons = cons
        self.pubr = pubr
        self.dmpr = dmpr
        self.emtr = emtr

    def __call__(self):
        self.cons.start() 
        self.pubr.start() 
        self.dmpr.start()
        self.emtr.start()

    def publish(self, obj):
        self.pubr.q.put(obj)

    @classmethod
    def cleanup(cls):
        print "cleanup : terminating active subprocess children ... "
        for c in multiprocessing.active_children():
            c.terminate()
         
    @classmethod
    def addr(cls):
        n = platform.node()
        p = multiprocessing.current_process()
        t = threading.current_thread()
        return "%s:%s:%s:%s" % ( n, p.name, p.pid, t.name )
 
    @classmethod
    def ls(cls):
        for c in multiprocessing.active_children():
            pika.log.info( "%r", c ) 
        for t in threading.enumerate():
            pika.log.info( "%r", t ) 
    
    def __repr__(self):
        return "\n".join( [ "cons %r" % self.cons , "pubr %r" % self.pubr , "dmpr %r" % self.dmpr , "emtr %r" % self.emtr, "" ] ) 



if __name__ == '__main__':


    pmq = PikaMQ(fail=False)
    pmq()    



