
Monitoring
===========

:google:`munin monit cacti nagios`


problems in dayabay context
----------------------------

#. do not want monitoring node to have the ssh keys to everything to be monitored

nagios
-------

#. nagios is venerable, apache + perl based, pain to configure, big community 

http://www.macworld.com/article/1134079/nagios.html

macports is at 3.2.3


others
-------

#. Monit, God, Supervisord, Upstart

   #. focus on starting/restarting daemons and services

#. Munin, Cacti

   #. focus on visualization of RRDTool data

#. Collectd

  #. focus on collecting and publishing data 


fabric-cuisine-watchdog/daemonwatch
--------------------------------------

Python based flexibility, more bare-bones : **more suitable to simple monitoring**

http://www.slideshare.net/confoo/server-administration-in-python-with-fabric-cuisine-and-watchdog

#. :google:`fabric cuisine watchdog`


#. fabric : python based ssh access to remote nodes, low level

    #. cuisine : simple function extensions using fabric primitives to add file/dir/text/user/group/sudo ops
    
        #. http://github.com/sebastien/cuisine  
        #. https://github.com/sebastien/cuisine/blob/master/src/cuisine.py  
        #. upstart is just based on ``service name status/restart``


    #. daemonwatch : (formerly watchdog)
    
        #. https://github.com/sebastien/daemonwatch  
        #. https://github.com/sebastien/daemonwatch/blob/master/Sources/daemonwatch.py
        #. service is a collection of rules, with a frequency associated
        #. rules can succeed or fail and have output
        #. actions are bound to rule, triggered on success or fail



#. i dont see the integration between daemonwatch and the others, daemonwatch looks to be entirely localnode


::

        #!/usr/bin/env python
        from watchdog import *

        send_email = Email( "name@whereever", "Subj", "confiug....")
        send_xmpp =  XMPP( "name@jabber", "Subj", "confiug....")

        Monitor(    # the "main"
          Service(      # Service monitors the rules
              name="...",
              monitor= (
                   HTTP(     # HTTP rule allows to test url
                      GET="http://...",
                      freq=Time.s(1),
                      timeout=Time.ms(80),
                      #fail=[
                      #    Print("Failed..."),send_email,send_xmpp,
                      #  ]      
              
                      fail = [
                          Incident( errors=5, during=Time.s(10), actions=[send_email,send_xmpp])
                             ]
                      )
                      )
                 )
               ).run()  

         # also Incident (smart action) to check if something happening repeatedly within time windows


