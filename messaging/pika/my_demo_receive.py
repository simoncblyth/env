#!/usr/bin/env python
'''
Example of simple consumer, waits one message, replies an ack and exits.
'''

import sys
import pika
import asyncore

from private import Private
p = Private()

conn = pika.AsyncoreConnection(pika.ConnectionParameters(p('AMQP_SERVER'),
        credentials = pika.PlainCredentials(p('AMQP_USER'), p('AMQP_PASSWORD')),
        heartbeat = 10))

print 'Connected to %r' % (conn.server_properties,)

qname = 'pika_demo_receive'

ch = conn.channel()
ch.queue_declare(queue=qname, durable=True, exclusive=False, auto_delete=False)
ch.queue_bind(   queue=qname,  exchange='abtdaq' , routing_key='abt.test.string' )


def handle_delivery(ch, method, header, body):
    print "method=%r" % (method,)
    print "header=%r" % (header,)
    print "  body=%r" % (body,)
    ch.basic_ack(delivery_tag = method.delivery_tag)

ch.basic_consume(handle_delivery, queue = qname)
pika.asyncore_loop()
print 'Close reason:', conn.connection_close
