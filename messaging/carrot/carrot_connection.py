# http://ask.github.com/carrot/introduction.html

from private import Private
p = Private()
params = { 'hostname':p('AMQP_SERVER'), 'port':p('AMQP_PORT'), 'userid':p('AMQP_USER'), 'password':p('AMQP_PASSWORD'), 'virtual_host':p('AMQP_VHOST') } 

from carrot.connection import BrokerConnection
conn = BrokerConnection( **params )

if __name__=='__main__':
    print params, conn

