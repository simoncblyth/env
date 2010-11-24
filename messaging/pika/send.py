#!/usr/bin/env python
'''
     pika-send    -v -s _CMS01
     pika-send    -v -s _CUHK

Assuming you have the corresponding private config variables 
in your $ENV_PRIVATE_PATH file
     AMQP_CMS01_SERVER
     AMQP_CMS01_USER
     AMQP_CMS01_PASSWORD


Example of simple producer, creates one message and exits.
'''

import sys
import pika
import asyncore
import platform

from private import Private
p = Private()

from optparse import OptionParser
op = OptionParser(usage=__doc__)
op.add_option("-k", "--routing-key")
op.add_option("-x", "--exchange")
op.add_option("-b", "--body")
op.add_option("-s", "--server")
op.add_option("-v", "--verbose" , action="store_true" )
op.set_defaults( verbose=False, routing_key="abt.test.string" , exchange="abt" , body="test message from %s on %s " % (sys.argv[0], platform.node() ))

def send( opts, args ):

    srv = opts.server or ""
    cfg = dict(server=p('AMQP%s_SERVER' % srv ), user=p('AMQP%s_USER' % srv ),  password=p('AMQP%s_PASSWORD' % srv ) )
    if opts.verbose:
        print opts, args
        print "config for server %s : %s " % ( srv , repr(cfg))

    conn = pika.AsyncoreConnection(pika.ConnectionParameters( cfg['server'] , credentials=pika.PlainCredentials( cfg['user'], cfg['password'] )))
    ch = conn.channel()

    props = pika.BasicProperties( content_type = "text/plain",  delivery_mode = 2, ) # persistent
    ch.basic_publish(
                 exchange=opts.exchange,routing_key=opts.routing_key,body=opts.body,
                 properties=props , block_on_flow_control = True)
    conn.close()
    pika.asyncore_loop()


if __name__=='__main__':
    send( *op.parse_args() )

