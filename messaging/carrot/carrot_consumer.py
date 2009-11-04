# http://ask.github.com/carrot/introduction.html


from optparse import OptionParser

from carrot_connection import conn, params 
from carrot.messaging import Consumer

def main(argv):
    usage = "usage: %prog [options] arg"
    parser = OptionParser(usage)
    parser.add_option("-q", "--queue",    dest="queue",   default="feed", help="name of the queue")
    parser.add_option("-e", "--exchange", dest="exchange", default="feed" , help="name of the exchange")
    parser.add_option("-k", "--key",      dest="key",      default="importer", help="name of the routing key")
    (opts, args) = parser.parse_args(argv)

    pars = { 'connection':conn , 'queue':opts.queue , 'exchange':opts.exchange , 'key':opts.key }
    print "creating consumer %s on connection %s " % ( repr(pars) , repr(params) )
    consumer = Consumer( **pars )
    def _callback(msg_data, msg):
        msg.__class__.__repr__ = lambda x:"< %s.%s object at 0x%x  ; %s %s %s %s >" % ( x.__class__.__module__, x.__class__.__name__, id(x), x.content_encoding , x.content_type , x.delivery_info , x.delivery_tag  )
        print "Got message : %s with data of length %s " % ( msg , len(msg_data) )
        if len(msg_data) < 200:print "msg_data: %s " % msg_data
        msg.ack()
    consumer.register_callback(_callback)
    print "enter consumer wait ... " 
    consumer.wait()  


if __name__=='__main__':
    import sys
    sys.exit(main(sys.argv))


