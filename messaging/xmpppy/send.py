import sys
import xmpp

from private import Private
p = Private()

class User(dict):
    def __init__(self, id ):
	self.update( id=id, host=p('EJABBERD_HOST_%d'%id), user=p('EJABBERD_USER_%d'%id), pass_=p('EJABBERD_PASS_%d'%id) )
    def jid(self):
        return "%s@%s" % ( self['user'], self['host'] ) 

fr = User(5) 
to = User(1)
msg = "  from %s to %s ... with %s " % ( fr , to , sys.argv[0] )


cl = xmpp.Client( p('EJABBERD_HOST') )
cl.connect(server=(fr['host'],p('EJABBERD_PORT')))
cl.auth( fr['user'] , fr['pass_'])
cl.send(xmpp.Message( to.jid() , msg ))


print msg 



