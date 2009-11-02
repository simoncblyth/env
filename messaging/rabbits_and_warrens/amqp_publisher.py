# http://blogs.digitar.com/jjww/2009/01/rabbits-and-warrens/

from amqplib import client_0_8 as amqp
from amqp_server import AMQPServer
v = AMQPServer.vhost()

import sys
msg = amqp.Message(sys.argv[1])
msg.properties["delivery_mode"] = 2

v.basic_publish(msg,exchange="sorting_room",routing_key="jason")
v.close()


