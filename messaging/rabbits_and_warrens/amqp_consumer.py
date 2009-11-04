# http://blogs.digitar.com/jjww/2009/01/rabbits-and-warrens/

from amqp_connection import AMQPConnection
v = AMQPConnection.vhost()
print v

v.exchange_declare(exchange="sorting_room", type="direct", durable=True, auto_delete=False,)
v.queue_declare(queue="po_box", durable=True, exclusive=False, auto_delete=False)
v.queue_bind(queue="po_box", exchange="sorting_room", routing_key="jason")

def recv_callback(msg):
    print 'Received: ' + msg.body + ' from channel #' + str(msg.channel.channel_id)

v.basic_consume(queue='po_box', no_ack=True, callback=recv_callback, consumer_tag="testtag")
while True:
    v.wait()
v.basic_cancel("testtag")

v.close()



