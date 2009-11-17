# http://ask.github.com/carrot/introduction.html


from ROOT import gSystem
gSystem.Load("$ENV_HOME/lib/libnotifymq.so")
from ROOT import MyTMessage
# hmm probably leave this on the C++/C side as cannot handle void* from pyroot ? 
# probably means cannot use the carrot consumer 

from optparse import OptionParser
from carrot_connection import conn, params 
from carrot.messaging import Consumer

def main(argv):
    usage = "usage: %prog [options] arg"
    parser = OptionParser(usage)
    parser.add_option("-q", "--queue",    dest="queue",   default="feed", help="name of the queue")
    parser.add_option("-e", "--exchange", dest="exchange", default="feed" , help="name of the exchange")
    parser.add_option("-k", "--key",      dest="key",      default="importer", help="name of the routing key")
    parser.add_option("-n", "--noack",    action="store_false" , dest="ack",    default=True , help="do not acknowledge the message")
    (opts, args) = parser.parse_args(argv)

    pars = { 'connection':conn , 'queue':opts.queue , 'exchange':opts.exchange , 'key':opts.key }   ## is the duplication of queue and exchange needed ?
    print "creating consumer %s on connection %s " % ( repr(pars) , repr(params) )
    consumer = Consumer( **pars )
    def _callback(msg_data, msg):
        msg.__class__.__repr__ = lambda x:"< %s.%s object at 0x%x  ; %s %s %s %s >" % ( x.__class__.__module__, x.__class__.__name__, id(x), x.content_encoding , x.content_type , x.delivery_info , x.delivery_tag  )
        print "Got message : %s with data of length %s " % ( msg , len(msg_data) )
        if len(msg_data) < 1000:print "msg_data: %s " % msg_data
        if opts.ack:
            print "acknowledging msg  "
            msg.ack()
        else:
            print "not acknowledging due to noack option "
    consumer.register_callback(_callback)
    print "enter consumer wait ... " 
    consumer.wait()  


if __name__=='__main__':
    import sys
    sys.exit(main(sys.argv))


