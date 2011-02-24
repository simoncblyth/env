#!/usr/bin/env python
import os
import pika
from ConfigParser import ConfigParser

class PikaClient(object):
    def __init__(self, opts):

        self.opts = opts
        self.pending = ['hello','world']
        self.messages = list()

        self.connection = None
        self.channel = None

    def properties(self, **kwargs):
        return pika.BasicProperties( **kwargs )

    def launch(self):
        self.connect()
        self.connection.ioloop.start()

    def connect(self):
        credentials = pika.PlainCredentials(self.opts.username, self.opts.password)
        param = pika.ConnectionParameters(host=self.opts.host, port=int(self.opts.port), virtual_host=self.opts.vhost , credentials=credentials)
        pika.log.info('PikaClient: Connecting to RabbitMQ with param %r', param )
        self.connection = pika.SelectConnection(param, on_open_callback=self.on_connected)
        self.connection.add_on_close_callback(self.on_closed)

    def on_connected(self, connection):
        pika.log.info('PikaClient: Connected : %r ', connection )
        self.connection = connection
        self.connection.channel(self.on_channel_open)

    def on_channel_open(self, channel):
        pika.log.info('PikaClient: channel_open : %r , Declaring Exchange', channel )
        self.channel = channel
        self.channel.exchange_declare(exchange=self.opts.exchange_name,
                                      type=self.opts.exchange_type ,
                                      auto_delete=self.opts.auto_delete,
                                      durable=self.opts.durable,
                                      callback=self.on_exchange_declared)

    def on_exchange_declared(self, frame):
        pika.log.info('PikaClient: Exchange Declared %r , Declaring Queue', frame )
        self.channel.queue_declare(queue=self.opts.queue_name,
                                   auto_delete=self.opts.auto_delete,
                                   durable=self.opts.durable,
                                   exclusive=self.opts.exclusive,
                                   callback=self.on_queue_declared)

    def on_queue_declared(self, frame):
        pika.log.info('PikaClient: Queue Declared %r , Binding Queue', frame )
        self.channel.queue_bind(exchange=self.opts.exchange_name,
                                queue=self.opts.queue_name,
                                routing_key=self.opts.routing_key,
                                callback=self.on_queue_bound)

    def on_queue_bound(self, frame):
        pika.log.info('PikaClient: Queue Bound %r, Issuing Basic Consume', frame )
        self.channel.basic_consume(consumer_callback=self.on_pika_message,
                                   queue=self.opts.queue_name,
                                   no_ack=self.opts.no_ack)
        for body in self.pending:
            self.channel.basic_publish(exchange=self.opts.exchange_name,
                                       routing_key=self.opts.routing_key ,
                                       body=body,
                                       properties=self.properties(content_type='text/plain', delivery_mode=1 ))

    def on_pika_message(self, channel, method, header, body):
        pika.log.info('PikaCient: Message receive, delivery tag #%i  body %r ', method.delivery_tag, body )
        self.messages.append(body)

    def on_basic_cancel(self, frame):
        pika.log.info('PikaClient: Basic Cancel Ok')
        self.connection.close()

    def on_closed(self, connection):
        self.connection.ioloop.stop()

    def sample_message(self, body='dummy sample msg'):
        self.channel.basic_publish(exchange=self.opts.exchange_name,
                                   routing_key=self.opts.routing_key ,
                                   body=body,
                                   properties=self.properties(content_type='text/plain', delivery_mode=1 ))

    def get_messages(self):
        output = self.messages
        self.messages = list()
        return output


class PikaOpts(dict):
    defaults = dict( 
                  host='127.0.0.1', username='guest', password='guest' , vhost='/' , port='5672' ,
                  queue_name='pmq' , exchange_name="pmx" , exchange_type="topic" , routing_key="pmq.#" , durable=True, exclusive=False, auto_delete=False, no_ack=False  
                  )
    def __getattr__(self, name ):
        return self.get(name, self.defaults.get(name, None)) 

    def read(self, path='~/.rmq.cnf', cnf='local' ):
        cfp = ConfigParser()
        cfp.read( os.path.expanduser(path) )
        assert cfp.has_section(cnf), ( cnf, cfp.sections() )
        for o in cfp.options( cnf ):   
            self[o] = cfp.get( cnf, o )



if __name__ == '__main__':

    pika.log.setup(color=True)

    po = PikaOpts()
    po.read(  path='~/.rmq.cnf' , cnf='pikamq' )

    pika.log.info( "PikaOpts %r ", po )

    pc = PikaClient(po)
    pc.launch()


