#!/usr/bin/env python
'''
Attempt to introduce threading to the pika examples
BUT failed to doit with threads alone, so using 
multiprocessing to workaround pika problems with 
consumer and publisher in the same process
    
    * http://docs.python.org/library/multiprocessing.html#module-multiprocessing 
    * http://www.ibm.com/developerworks/aix/library/au-multiprocessing/
    * https://github.com/tonyg/pika#readme
    * http://tonyg.github.com/pika/
    * http://www.amqp.org/confluence/download/attachments/720900/amqp0-9-1.pdf
    * http://www.rabbitmq.com/tutorial-three-python.html

Notes 
   * when changing parameters of an exchnage/queue do a rabbitmq-reset to remove 
     preexisting entities otherwise the callbacks never gets back to you 

   Declaring exchange: server channel error 406, message: PRECONDITION_FAILED - cannot redeclare exchange 'abt' in vhost '/' 
     with different type, durable, internal or autodelete value


Test with :
   pika- ; pika-i
   ipython> run mon.py

Publish from another process with :

   (mq)simon:pika blyth$ camqadm
    /Users/blyth/v/mq/lib/python2.6/site-packages/celery-2.2.4-py2.6.egg/celery/loaders/default.py:54: NotConfigured: No 'celeryconfig' module found! Please make sure it exists and is available to Python.
  "is available to Python." % (configname, )))
   -> connecting to amqplib://guest@localhost:5672/.
   -> connected.
   1> basic.publish aaa xmonf test
   ok.
   2> basic.publish bbb xmonf test
   ok.
   3> basic.publish ccc xmonf test
   ok.
   4> 

to start from a clean slate :

   sudo rabbitmqctl stop_app
   sudo rabbitmqctl reset   
   sudo rabbitmqctl start_app

ISSUES...

  1) duplication / round-robinning  
     getting messages twice and missing every other ???
 
     caused by double connected consumer
     which arose due to a one_shot=False for the default pika on_open callback ... 

     after fix this ... still failing to get a publisher and consumer to 
     co-exist within the same process  

NEXT

     * breaking up the monolith for reuse/flexibility 
          * splitting ROOT and pika dependencies
     * aping the rootmq gMQ API
     * distributed testing
     * GUI integration / timers
     * 


'''


import ROOT
ROOT.gSystem.Load("libAbtDataModel")
from aberdeen.DataModel.tests.evs import Evs


from ConfigParser import ConfigParser
import os, sys, platform, time, datetime, threading, multiprocessing, Queue, pickle, atexit
import pika

class Msg(dict):
    content_type = property( lambda self:self.get('content_type', 'text/pickle' ))
    def _body(self):
        b = self.get('body', "dummy-msg-%s" % datetime.datetime.now().strftime("%c") )
        return pickle.dumps( b )
    body = property( _body )


#class Worker(threading.Thread):
class Worker(multiprocessing.Process):

    is_thread_worker = property( lambda self:issubclass(self.__class__, threading.Thread ) )
    is_process_worker = property( lambda self:issubclass(self.__class__, multiprocessing.Process ) )

    def _tag(self): 
        if self.is_thread_worker:
             return "T%s" % self.name
        elif self.is_process_worker:
             return "P%s" % self.pid
        else:
             return "?"
    tag = property( _tag ) 


    def __init__(self, q, **kwargs ):
        #threading.Thread.__init__(self, name=kwargs['name'] )
        multiprocessing.Process.__init__(self, name=kwargs['name'])

        self.q = q
        self.opts = kwargs

        self.count = 0
        self.last_count = None
        self.last_time = None
        self.channel  = None
        self.connection = None


    def run(self):                        
        """ 
        Runs in separate thread once started 

        Note the low level way of setting the on_open callback in order to specify one_shot=True 
        otherwise the publisher ends up behaving as a 2nd consumer ... leading to confusion/roundrobining/duplication 

        """ 
        pika.log.debug("%s : starting " , self.tag )
        credentials = pika.PlainCredentials(self.opts['username'], self.opts['password'])
        parameters = pika.ConnectionParameters(self.opts['host'], credentials=credentials)
        connection = pika.SelectConnection( parameters )

        connection.callbacks.add(0, '_on_connection_open', self._on_connected() , one_shot=True )
        connection.add_on_close_callback( self._on_closed() )
        self.connection = connection
        connection.ioloop.start()

    def __call__(self, daemon=True):
        """Using all daemon threads allows the MainThread to exit while the daemons are still alive """
        if  issubclass(self.__class__, threading.Thread ): 
            self.daemon = daemon
        self.start()
 
    def _on_connected(self):
        def on_connected(connection):
            pika.log.info("%s: connected : %r " % ( self.tag , connection) )
            connection.channel(self._on_channel_open() )
            self.connection = connection
        return on_connected  

    def _on_channel_open(self):
        def on_channel_open(channel):
            pika.log.info("%s: channel_open : %r " % ( self.tag , channel )  )
            self.channel = channel
            args = dict( type=self.opts['type'], exchange=self.opts['exchange'], durable=self.opts['durable'], auto_delete=self.opts['auto_delete'], callback=self._on_exchange_declared() )
            pika.log.info("%s: exchange_declare start : %r ", self.tag, args )
            channel.exchange_declare( **args )
        return on_channel_open

    def _on_closed(self):
        def on_closed(connection):
            pika.log.info("%s: closing connection : %r " % ( self.tag , connection) )
            connection.ioloop.stop()
        return on_closed

    def close(self):
        connection = self.connection
        connection.close() 
        # Loop until we're fully closed, will stop on its own
        connection.ioloop.start()

    def _on_exchange_declared(self):
        if self.name == "CONSUMER":
            def on_exchange_declared(frame): 
                pika.log.info("%s: exchange_declared : %r " % ( self.tag , frame )  )
                channel = self.channel
                channel.queue_declare( queue=self.opts['queue'], durable=self.opts['durable'], exclusive=self.opts['exclusive'], auto_delete=self.opts['auto_delete'], callback=self._on_queue_declared() )
        else:
            def on_exchange_declared(frame): 
                pika.log.info("%s: exchange_declared : %r " % ( self.tag , frame )  )
                channel = self.channel
                while True:
                    obj = self.q.get()  # blocks 
                    if issubclass(obj.__class__, Msg ):
                        properties=pika.BasicProperties( content_type=obj.content_type, delivery_mode=1) 
                    else:
                        properties=pika.BasicProperties( content_type="text/plain", delivery_mode=1 )
                    body = obj
                    pika.log.info( "%s %r ", self.tag , body ) 
                    channel.basic_publish( exchange=self.opts['exchange'], routing_key=self.opts['routing_key'] , body=body, properties=properties ) 
        return on_exchange_declared

    def _on_queue_declared(self):
        def on_queue_declared(frame): 
            pika.log.info("%s: queue_declared : %r " % ( self.tag, frame )  )
            channel = self.channel
            channel.queue_bind( queue=self.opts['queue'], exchange=self.opts['exchange'], routing_key=self.opts['routing_key'], callback=self._on_ready() )
        return on_queue_declared

    def _on_ready(self):
        def on_ready(frame):
            pika.log.info("%s: ready : %r " % ( self.tag, frame ) )
            self.start_time = time.time()
            channel = self.channel
            channel.basic_consume( self._on_handle_delivery() , queue=self.opts['queue'] , no_ack=self.opts['no_ack'])
        return on_ready

    def _on_handle_delivery(self):  
        def on_handle_delivery(channel, method_frame, header_frame, body):
            pika.log.info(" %s : handle_delivery : channel %r  method %r header %r body %r " , self.tag, channel, method_frame, header_frame, body )
            pika.log.info(" %s : content_type %s delivery_tag %s " , self.tag, header_frame.content_type, method_frame.delivery_tag )
            self.q.put( body )
            if self.opts['no_ack']:
                pass
            else:
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
        return on_handle_delivery 



class Dumper(threading.Thread):          
    """
    Placing onto the pub queue is immediately dumped ... 
        q.put("helloq")

    BUT going via the rabbit .... 
    """
    def __init__(self, q , **kwargs):
        self.q = q 
        threading.Thread.__init__(self, name=kwargs['name'] )

    tag = property( lambda self:"%s:%s" % ( self.__class__.__name__, self.name ))

    def run(self):             
        pika.log.debug( "%s starting " , self.tag )
        while True:
            m = self.q.get()  # blocks 
            pika.log.info( "%s %r ", self.tag, m ) 

    def __call__(self, daemon=True):
        self.daemon = daemon
        self.start()
   

class Emitter(threading.Thread):          
    """
    Placing strings onto the argument q 
    every interval seconds ...

    Usage::

        emit = Emitter( q, interval=10 )
        emit()

    """
    def __init__(self, q , **kwargs):
        self.q = q 
        self.interval = kwargs.get('interval',1)
        threading.Thread.__init__(self, name=kwargs.get('name',self.__class__.__name__ )  )

    tag = property( lambda self:"%s:%s" % ( self.__class__.__name__, self.name ))

    def run(self):             
        while True:
            now = datetime.datetime.now() 
            msg = "%s-%s" % ( self.tag, now.strftime("%c") )
            self.q.put( msg )  
            time.sleep(self.interval)

    def __call__(self, daemon=True):
        self.daemon = daemon
        self.start()



 
 
def tls():
    for c in multiprocessing.active_children():
        pika.log.info( "%r", c ) 
    for t in threading.enumerate():
        pika.log.info( "%r", t ) 
 

def addr():
    n = platform.node()
    p = multiprocessing.current_process()
    t = threading.current_thread()
    return "%s:%s:%s:%s" % ( n, p.name, p.pid, t.name )

def cleanup():
    print "cleanup : terminating active subprocess children ... "
    for c in multiprocessing.active_children():
        c.terminate()


if __name__ == '__main__':

    atexit.register( cleanup )

    cnf = os.environ.get('RMQ_CONF','local')
    cfp = ConfigParser()
    cfp.read( os.path.expanduser("~/.rmq.cnf") )
    assert cfp.has_section(cnf), ( cnf, cfp.sections() )

    pika.log.setup(color=True, level=pika.log.INFO )    
    
    defaults = dict( server='127.0.0.1', username='guest', password='guest' )
    defaults.update( queue=addr(), exchange="abt" , type="topic" , routing_key="abt.#" , durable=True, exclusive=False, auto_delete=False, no_ack=False  )

    cfg = defaults
    for o in cfp.options( cnf ):   
        cfg[o] = cfp.get( cnf, o )

    cfg['host'] = cfg.pop('server')
    pika.log.info( "cnf %s cfg %r ", cnf, cfg ) 

    ## sender and receiver queues 
   
    q = multiprocessing.Queue()   
    cons = Worker( q , **dict( cfg, name="CONSUMER" )  )    ## updates q as messages arrive
    cons()    

    p = multiprocessing.Queue()   
    pubr = Worker( p , **dict( cfg, name="PUBLISHER" ) )    ## put to p to send messages
    pubr()
     
    ## start subprocesses before threads to avoid forking the threads     

    dmpr = Dumper( q , name="DUMPER" )
    dmpr()                                                  ## thread to dump messages as they arrive in local q 

    emit = Emitter( p, interval=10 )
    emit()

    tls()

    ## sending pickled ROOT objects, working OK 
    evs = Evs()
    p.put( pickle.dumps(evs.ri) )
    p.put( pickle.dumps(evs[0]) )
    p.put( pickle.dumps(evs[1]) )




