# http://ask.github.com/carrot/introduction.html

from carrot_connection import conn, params 
from carrot.messaging import Consumer
consumer = Consumer(connection=conn, queue="feed", exchange="feed", routing_key="import_pictures")

msg = None
msgd = None

def import_picture_callback(message_data, message):
    print "import_picture_callback ", message
    # allows ipython interrogation after interrupt out of the wait 
    global msgd, msg    
    msgd, msg = message_data, message 
    msg.__class__.__repr__ = lambda x:"< %s.%s object at 0x%x  ; %s %s %s %s >" % ( x.__class__.__module__, x.__class__.__name__, id(x), x.content_encoding , x.content_type , x.delivery_info , x.delivery_tag  )
    print msg
    print " %s %s %s " % ( len(message_data), len(message.payload), len(message.body) )  ## why all this duplication ?
    assert msgd[0:3]  ==  '\xff\xd8\xff' , "binary data appears not to be a jpeg "
    message.ack()

consumer.register_callback(import_picture_callback)

print "enter binary consumer loop ... %s " % repr(params)
consumer.wait() # Go into the consumer loop.

    



