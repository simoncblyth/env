#!/usr/bin/env python
'''
Attempt to introduce threading to the pika examples

    * https://github.com/tonyg/pika#readme
    * http://tonyg.github.com/pika/
    * http://www.amqp.org/confluence/download/attachments/720900/amqp0-9-1.pdf

    * http://www.rabbitmq.com/tutorial-three-python.html
 

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
    try to use multiprocessing to workaround pika problems with 
    consumer and publisher in the same process

        http://docs.python.org/library/multiprocessing.html#module-multiprocessing 
        http://www.ibm.com/developerworks/aix/library/au-multiprocessing/
   from multiprocessing import Process, Queue



'''

import os, sys, platform, time, threading
from Queue import Queue
import pika
from pika.adapters import SelectConnection


class Worker(threading.Thread):
    nam = property( lambda self:"%s:%s" % ( self.__class__.__name__, self.getName() ))
    def __init__(self, q, **kwargs ):
        threading.Thread.__init__(self)

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
        pika.log.debug("%s : starting " , self.nam )
        credentials = pika.PlainCredentials(self.opts['username'], self.opts['password'])
        parameters = pika.ConnectionParameters(self.opts['host'], credentials=credentials)
        connection = SelectConnection( parameters )

        connection.callbacks.add(0, '_on_connection_open', self._on_connected() , one_shot=True )
        connection.add_on_close_callback( self._on_closed() )
        self.connection = connection
        connection.ioloop.start()

    def __call__(self, daemon=False):
        """Using all daemon threads allows the MainThread to exit while the daemons are still alive """
        self.daemon = daemon
        self.start()
 
    def _on_connected(self):
        def on_connected(connection):
            pika.log.info("%s: connected : %r " % ( self.nam , connection) )
            connection.channel(self._on_channel_open() )
            self.connection = connection
        return on_connected  

    def _on_channel_open(self):
        def on_channel_open(channel):
            pika.log.info("%s: channel_open : %r " % ( self.nam , channel )  )
            self.channel = channel
            channel.exchange_declare( type=self.opts['type'], exchange=self.opts['exchange'], durable=self.opts['durable'], auto_delete=self.opts['auto_delete'], callback=self._on_exchange_declared() )
        return on_channel_open

    def _on_closed(self):
        def on_closed(connection):
            pika.log.info("%s: closing connection : %r " % ( self.nam , connection) )
            connection.ioloop.stop()
        return on_closed

    def close(self):
        connection = self.connection
        connection.close() 
        # Loop until we're fully closed, will stop on its own
        connection.ioloop.start()

    def _on_exchange_declared(self):
        if self.opts['mode'] == "CONSUMER":
            def on_exchange_declared(frame): 
                pika.log.info("%s: exchange_declared : %r " % ( self.nam , frame )  )
                channel = self.channel
                channel.queue_declare( queue=self.opts['queue'], durable=self.opts['durable'], exclusive=self.opts['exclusive'], auto_delete=self.opts['auto_delete'], callback=self._on_queue_declared() )
        else:
            def on_exchange_declared(frame): 
                channel = self.channel
                while True:
                    m = self.q.get()  # blocks 
                    pika.log.info( "%s %r ", self.nam, m ) 
                    channel.basic_publish(exchange=self.opts['exchange'], routing_key=self.opts['routing_key'],
                              body=m, properties=pika.BasicProperties( content_type="text/plain", delivery_mode=1))
        return on_exchange_declared

    def _on_queue_declared(self):
        def on_queue_declared(frame): 
            pika.log.info("%s: queue_declared : %r " % ( self.nam , frame )  )
            channel = self.channel
            channel.queue_bind( queue=self.opts['queue'], exchange=self.opts['exchange'], routing_key=self.opts['routing_key'], callback=self._on_ready() )
        return on_queue_declared

    def _on_ready(self):
        def on_ready(frame):
            pika.log.info("%s: ready : %r " % ( self.nam, frame ) )
            self.start_time = time.time()
            channel = self.channel
            channel.basic_consume( self._on_handle_delivery() , queue=self.opts['queue'] , no_ack=self.opts['no_ack'])
        return on_ready

    def _on_handle_delivery(self):  
        def on_handle_delivery(channel, method_frame, header_frame, body):
            pika.log.debug(" %s : handle_delivery : channel %r  method %r header %r body %r " , self.nam, channel, method_frame, header_frame, body )
            pika.log.debug(" %s : content_type %s delivery_tag %s " , self.nam, header_frame.content_type, method_frame.delivery_tag )
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
 
            pika.log.info(" %s : timed_receive: %i Messages Received, %.4f per second", self.nam, self.count, rate)
        return on_handle_delivery 



class Dumper(threading.Thread):          
    """
    Placing onto the pub queue is immediately dumped ... 
        q.put("helloq")

    BUT going via the rabbit .... 
    """
    def __init__(self, q ):
        self.q = q 
        threading.Thread.__init__(self)

    nam = property( lambda self:"%s:%s" % ( self.__class__.__name__, self.getName() ))

    def run(self):             
        pika.log.debug( "%s starting " , self.nam )
        while True:
            m = self.q.get()  # blocks 
            pika.log.info( "%s %r ", self.nam, m ) 

    def __call__(self, daemon=False):
        self.daemon = daemon
        self.start()
   


 
def tls():
    for t in threading.enumerate():
        pika.log.info( "%r", t ) 





if __name__ == '__main__':

    pika.log.setup(color=True, level=pika.log.INFO )    


    qn = "%s-%s-%s" % ( platform.node(), os.getpid(), threading.current_thread().getName() )
    cfg = dict( queue=qn , routing_key="test", exchange="xmonf", type="fanout", durable=False , exclusive=False, auto_delete=True, host='127.0.0.1' , username='guest', password='guest', no_ack=True )
    

    q = Queue()        ## receiver
    cons = Worker( q , **dict( cfg, mode="CONSUMER" )  )
    cons(daemon=True)    ## start thread to update the local q as messages arrive from remote rabbitmq q          

    dmpr = Dumper( q )
    dmpr(daemon=True)    ## start thread to dump messages as they arrive in local q 



    #p = Queue()   ## sender
    #pubr = Worker( p , **dict( cfg, mode="PUBLISHER" ) )
    #pubr(daemon=True)


    tls()


    ##
    ##     ... only every other message gets thru 
    ##                  ... roundrobin-ing? 
    ##                  ... change q names ???
    ##

    #time.sleep(1)



