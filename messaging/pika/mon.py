#!/usr/bin/env python
'''
Attempt to introduce threading to the pika examples

Hookup to virtual python 
    . ~/v/mq/bin/activate

The virtual python should be equipped with :
      * ipython  ... pip install --upgrade ipython
              /Users/blyth/v/mq/bin/ipython      
                     CAUTION USE FULL PATH TO IPYTHON 

      * camaqdm  ... from celery 

             camqadm-
             camqadm--


Test with 

   pika-
   pika-mon  
   pika-imon


PECULIARITIUES ...

  with no_ack=True...

       Getting 20 msgs received when expect 10 
          *  or i need to do cleaner exits ... killing window at moment 
          * unhealthy blank exchange ??

  camqadm kills the consumer with ...


mq)simon:e blyth$ camqadm
/Users/blyth/v/mq/lib/python2.6/site-packages/celery-2.2.4-py2.6.egg/celery/loaders/default.py:54: NotConfigured: No 'celeryconfig' module found! Please make sure it exists and is available to Python.
  "is available to Python." % (configname, )))
-> connecting to amqplib://guest@localhost:5672/.
-> connected.
2> basic.publish a "" test
ok.


3> basic.publish a "" test
ok.
4> basic.publish b "" test
ok.


    .... but the toubles seem not to happen if do not try to produce and consume
    from the same process (in different threads)
   
     ...... smth is being shared that should not be ????


'''
import pika
import sys
import time
import threading

from pika.adapters import SelectConnection

pika.log.setup(color=True)

class MQThread(threading.Thread):          
    def __init__(self, **opts ):
        self.channel  = None
        self.connection = None
        self.opts = opts      
        threading.Thread.__init__(self)

    nam = property( lambda self:self.__class__.__name__ )

    def run(self):                        
        pika.log.info("%s : MQThread run " , self.nam )
        parameters = pika.ConnectionParameters(self.opts['host'])
        connection = SelectConnection(parameters, self._on_connected() )
        self.connection = connection
        connection.ioloop.start()

    def _on_connected(self):
        def on_connected(connection):
            pika.log.info("%s: connected : %r " % ( self.nam , connection) )
            connection.channel(self._on_channel_open() )
            self.connection = connection
        return on_connected  

    def _on_channel_open(self):
        def on_channel_open(channel_):
            channel = channel_
            pika.log.info("%s: channel_open : %r " % ( self.nam , channel_ )  )
            channel.queue_declare( queue=self.opts['queue'], durable=self.opts['durable'], exclusive=self.opts['exclusive'], auto_delete=self.opts['auto_delete'], callback=self._on_queue_declared() )
            self.channel = channel
        return on_channel_open

    def close(self):
        self.connection.close() 




class Consumer(MQThread):
    def __init__(self, **kwargs ):
        MQThread.__init__(self, **kwargs )
        self.count = 0
        self.last_count = None
        self.last_time = None

    def _on_queue_declared(self):
        def on_queue_declared(frame):
            pika.log.info("%s: queue_declared : %r " % ( self.nam, frame ) )
            self.start_time = time.time()
            channel = self.channel
            channel.basic_consume( self._on_handle_delivery() , queue=self.opts['queue'] , no_ack=self.opts['no_ack'])
        return on_queue_declared 

    def _on_handle_delivery(self):  
        def on_handle_delivery(channel, method_frame, header_frame, body):
            pika.log.info(" %s : handle_delivery : channel %r  method %r header %r body %r " , self.nam, channel, method_frame, header_frame, body )
            pika.log.info(" %s : content_type %s delivery_tag %s " , self.nam, header_frame.content_type, method_frame.delivery_tag )

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




class ProduceN(MQThread):
    """
        Sends n messages then disconnects
    """
    def __init__(self, **kwargs ):
        MQThread.__init__(self, **kwargs )
 
    def _on_queue_declared(self):
        def on_queue_declared(frame):
            pika.log.info("%s: queue_declared : %r " % ( self.nam, frame ) )
            self.start_time = time.time()
            channel = self.channel
            connection = self.connection

            for n in range(self.opts['n']):
                channel.basic_publish(exchange=self.opts['exchange'], routing_key=self.opts['routing_key'],
                              body=self.opts['body'] % n, properties=pika.BasicProperties( content_type="text/plain", delivery_mode=1))
            connection.close()
        return on_queue_declared 






if __name__ == '__main__':

    cfg = dict( queue='test' , durable=True, exclusive=False, auto_delete=False, modo=0 , host='127.0.0.1' , no_ack=True )

    cons = Consumer( **cfg  )
    cons.start()           
    pika.log.info("started the consumer thread ")

    #time.sleep(1)

    #prod = ProduceN( **dict(cfg, routing_key="test" , body="testing-message-body-%i" , n=10  , exchange='' ) ) 
    #prod.start()
    #pika.log.info("started the produceN thread ")



