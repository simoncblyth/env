#!/usr/bin/env python
"""
From the spec... topic exchange :
  * message queue binds to the exchange using a routing pattern
  * publisher sends the exchange a message with routing key R
  * message is passed to the message queue if R matches P

The pattern... [a-z][A-Z][0-9]  words separated by dots...
  * '''*''' matches a single word
  * '''#''' matches zero or more words
"""
import sys
import pika
import asyncore
import platform
from optparse import OptionParser

op = OptionParser()
op.add_option("-q", "--queue")
op.add_option("-x", "--exchange")
op.add_option("-k", "--routing-key")

op.set_defaults(
     queue="%s_%s" % ( sys.argv[0] , platform.node() ), 
     routing_key="demo.routing.key" , 
     exchange="amq.topic" , 
)

def handle_delivery(ch, method, header, body):
    print "method=%r" % (method,)
    print "header=%r" % (header,)
    print "  body=%r" % (body,)
    ch.basic_ack(delivery_tag = method.delivery_tag)

def consume( opts, args ):
    print opts, args

    from private import Private
    p = Private()
    conn = pika.AsyncoreConnection(pika.ConnectionParameters(p('AMQP_SERVER'),
              credentials = pika.PlainCredentials(p('AMQP_USER'), p('AMQP_PASSWORD')),
              heartbeat = 10))

    print 'Connected to %r' % (conn.server_properties,)

    ch = conn.channel()
    ch.queue_declare(queue=opts.queue , durable=False , exclusive=False, auto_delete=False)
    ch.queue_bind(   queue=opts.queue , exchange=opts.exchange  , routing_key=opts.routing_key )
    ch.basic_consume( handle_delivery, queue = opts.queue )
    pika.asyncore_loop()
    print 'Close reason:', conn.connection_close


if __name__=='__main__':
    consume(*op.parse_args())


