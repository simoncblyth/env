# http://blogs.digitar.com/jjww/2009/01/rabbits-and-warrens/

from amqplib import client_0_8 as amqp
from amqp_connection import AMQPConnection
v = AMQPConnection.vhost()
print v

import sys
msg = amqp.Message(" ".join(sys.argv[1:]))
msg.properties["delivery_mode"] = 2

v.basic_publish(msg,exchange="sorting_room",routing_key="jason")
v.close()


