# http://blogs.digitar.com/jjww/2009/01/rabbits-and-warrens/

class AMQPServer(object):
    @classmethod
    def vhost(cls):
        from env.base.private import Private
        p = Private()
        v = AMQPServer( host="%s:%s" % ( p('AMQP_SERVER'), p('AMQP_PORT') ) , userid=p('AMQP_USER'), password=p('AMQP_PASSWORD'), virtual_host=p('AMQP_VHOST'), insist=False )   
        return v
    def __init__(self, **kwa):
        from amqplib import client_0_8 as amqp
        conn = amqp.Connection( **kwa )
        chan = conn.channel()
        self.conn = conn
        self.chan = chan 
    def close(self):
        self.chan.close()
        self.conn.close()

    def __getattribute__(self, name ):
        """  methods not implemented in self are passed on to the channel """
        try:
            return object.__getattribute__(self, name)
        except AttributeError:
            return getattr( self.chan , name )


if __name__=='__main__':
    from amqp_server import AMQPServer
    v = AMQPServer.vhost() 
    print v




