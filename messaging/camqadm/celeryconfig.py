"""
   
   Propagate private config settings into the
      BROKER_* variables needed by celery 

   And in particular camqadm 
      http://ask.github.com/celery/reference/celery.bin.camqadm.html 


   Switch default server by eg 

      local AMQP_DEFAULT=_CMS01
      local AMQP_DEFAULT=_CUHK

   Assuming corresponding  configsets
      local AMQP_CMS01_SERVER=...   etc..

"""

from private import Private
p = Private()

srv = p('AMQP_DEFAULT') or ""
req = dict( server='AMQP%s_SERVER' % srv , user='AMQP%s_USER' % srv,  password='AMQP%s_PASSWORD' % srv )
cfg = p( **req )
cfg = dict( cfg , vhost="/", port="5672" )

print "using srv %s ... try \"help\" " % srv

## Broker settings.

BROKER_HOST = cfg["server"]
BROKER_PORT = cfg["port"]
BROKER_VHOST = cfg["vhost"]
BROKER_USER = cfg["user"]
BROKER_PASSWORD = cfg["password"]



