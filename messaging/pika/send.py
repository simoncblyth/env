#!/usr/bin/env python
'''
Example of simple producer, creates one message and exits.
'''

import sys
import pika
import asyncore
import platform

from private import Private
p = Private()

from optparse import OptionParser
op = OptionParser()
op.add_option("-k", "--routing-key")
op.add_option("-x", "--exchange")
op.add_option("-b", "--body")
op.set_defaults( routing_key="demo.routing.key" , exchange="amq.topic" , body="test message from %s on %s " % (sys.argv[0], platform.node() ))

def send( opts, args ):
    print opts, args
    conn = pika.AsyncoreConnection(pika.ConnectionParameters( p('AMQP_SERVER'),
         credentials=pika.PlainCredentials(p('AMQP_USER'), p('AMQP_PASSWORD'))))
    ch = conn.channel()

    props = pika.BasicProperties( content_type = "text/plain",  delivery_mode = 2, ) # persistent
    ch.basic_publish(
                 exchange=opts.exchange,routing_key=opts.routing_key,body=opts.body,
                 properties=props , block_on_flow_control = True)
    conn.close()
    pika.asyncore_loop()


if __name__=='__main__':
    send( *op.parse_args() )

