from private import Private
from bunny import Bunny
from amqplib import client_0_8 as amqp

def _help_private_connect(self):
    print "\n".join(["\tprivate_connect <name>",
                     "\tConnect to vhost in AMQP server specified by private- config vars : ",
                     "\t    AMQP_<name>SERVER/USER/PASSWORD/VHOST ",
                     "\tthe single parameter <name> defaults to an empty string.",
                    ])

def _do_private_connect(self, name='' ):
    p = Private()
    host = p("AMQP_%sSERVER" % name )
    user = p("AMQP_%sUSER" % name )
    password = p("AMQP_%sPASSWORD" % name )
    vhost = p("AMQP_%sVHOST" % name )
    try:
      print "Trying connect to %s:%s as %s ... name:%s " % (host, vhost, user, name)
      self.conn = amqp.Connection(userid=user, password=password, host=host, virtual_host=vhost, ssl=False)
      self.chan = self.conn.channel()
      print "Success!"
      """connection/channel creation success, change prompt"""
      self.prompt = "%s.%s: " % (host, vhost)
    except Exception , out:
      print "Connection or channel creation failed"
      print "Error was: ", out

Bunny.do_private_connect = _do_private_connect
Bunny.help_private_connect = _help_private_connect

if __name__=='__main__':
    shell = Bunny()
    import sys
    if len(sys.argv) == 2:
        name = sys.argv[1]
    else:
        name = ''
    shell.do_private_connect(name)
    shell.cmdloop()



