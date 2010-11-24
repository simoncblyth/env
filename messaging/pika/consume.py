#!/usr/bin/env python
"""
     pika-consume -v -s _CMS01
     pika-consume -v -s _CUHK

Assuming you have the corresponding private config variables 
in your $ENV_PRIVATE_PATH file
     AMQP_CMS01_SERVER
     AMQP_CMS01_USER
     AMQP_CMS01_PASSWORD

Specify a default configset using eg: 
     AMQP_DEFAULT=_CMS01



From the spec... topic exchange :
  * message queue binds to the exchange using a routing pattern
  * publisher sends the exchange a message with routing key R
  * message is passed to the message queue if R matches P

The pattern... [a-z][A-Z][0-9]  words separated by dots...
  * '''*''' matches a single word
  * '''#''' matches zero or more words
"""
import sys
import os
import pika
import asyncore
import platform
from optparse import OptionParser

op = OptionParser(usage=__doc__)
op.add_option("-o", "--only-bind", action="store_true" , help="just for control of consume.py, not part of protocol"  ) 
op.add_option("-v", "--verbose", action="store_true" ) 
op.add_option("-s", "--server" , help="allow easy switching between private configs AMQP%(srv)s_SERVER/USER/PASSWORD eg with _CUHK " ) 

op.add_option("-q", "--queue")
op.add_option("-k", "--routing-key")
op.add_option("-x", "--exchange")

op.add_option("-p", "--passive",    action="store_true" ) 
op.add_option("-d", "--durable",   action="store_true" ) 
op.add_option("-e", "--exclusive", action="store_true" ) 
op.add_option("-a", "--auto-delete", action="store_true" ) 

op.set_defaults(
     queue="%s@%s" % ( os.path.basename(sys.argv[0]) , platform.node() ), 
     routing_key="#.string" , 
     exchange="abt" , 
     
     passive=False,
     durable=False,
     exclusive=False,
     auto_delete=False,

     only_bind=False,        ## just my flow control not part of AMQP
     verbose=False,
     server="",
)

# passive:no durable:no exclusive:no auto_delete:no 

def handle_delivery(ch, method, header, body):
    print "method=%r" % (method,)
    print "header=%r" % (header,)
    print "  body=%r" % (body,)
    ch.basic_ack(delivery_tag = method.delivery_tag)

def consume( opts, args ):
    from private import Private
    p = Private()
    
    srv = opts.server or p('AMQP_DEFAULT') or ""
    req = dict( server='AMQP%s_SERVER' % srv , user='AMQP%s_USER' % srv,  password='AMQP%s_PASSWORD' % srv )
    cfg = p( **req )

    if opts.verbose:
        print opts, args
        print "config for server %s : %s " % ( srv , repr(cfg))

    conn = pika.AsyncoreConnection(pika.ConnectionParameters( cfg['server'],
              credentials = pika.PlainCredentials( cfg['user'], cfg['password'] ),
              heartbeat = 10))
    print 'Connected to %r' % (conn.server_properties,)
    ch = conn.channel()

    queue_declare = dict(queue=opts.queue , durable=opts.durable , exclusive=opts.exclusive , auto_delete=opts.auto_delete , passive=opts.passive )
    print 'Queue declare %s ' % repr(queue_declare )
    ch.queue_declare( **queue_declare )

    queue_bind = dict( queue=opts.queue , exchange=opts.exchange  , routing_key=opts.routing_key )
    print 'Queue bind %s ' % repr(queue_bind )
    ch.queue_bind(  **queue_bind ) 

    if opts.only_bind:
        print "only binding... "
    else:
        print "basic_consume from queue %s ... " % opts.queue
        ch.basic_consume( handle_delivery, queue = opts.queue )
    pika.asyncore_loop()
    print 'Close reason:', conn.connection_close


if __name__=='__main__':
    consume(*op.parse_args())


