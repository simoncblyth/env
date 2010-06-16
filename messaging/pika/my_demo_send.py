#!/usr/bin/env python
'''
Example of simple producer, creates one message and exits.
'''

import sys
import pika
import asyncore

from private import Private
p = Private()

conn = pika.AsyncoreConnection(pika.ConnectionParameters( p('AMQP_SERVER'),
        credentials=pika.PlainCredentials(p('AMQP_USER'), p('AMQP_PASSWORD'))))

ch = conn.channel()

ch.basic_publish(exchange='abtdaq',
                 routing_key="abt.test.string",
                 body="Hello World! from pika basic_publish ",
                 properties=pika.BasicProperties(
                        content_type = "text/plain",
                        delivery_mode = 2, # persistent
                        ),
                 block_on_flow_control = True)

conn.close()
pika.asyncore_loop()
