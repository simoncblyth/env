#!/usr/bin/env python

import pika
import asyncore
from private import Private
p = Private()

exchange , key, body =  "abt", "test.key.string" , "test.body" 
def publish_to_q( exchange, key , body ):
    """
         

    """
    print "publish_to_q %s %s %s " % ( exchange, key, body )
    conn = pika.AsyncoreConnection(pika.ConnectionParameters( p('AMQP_SERVER'),credentials=pika.PlainCredentials(p('AMQP_USER'), p('AMQP_PASSWORD'))))
    ch = conn.channel()
    props = pika.BasicProperties( content_type = "text/plain",  delivery_mode = 2, ) # persistent
    ch.basic_publish( exchange = str(exchange), routing_key = str(key), body = str(body), properties = props , block_on_flow_control = True)
    conn.close()
    pika.asyncore_loop()


if __name__ == '__main__':
    pass

    

