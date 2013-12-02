# === func-gen- : network/dbus fgp network/dbus.bash fgn dbus fgh network
dbus-src(){      echo network/dbus.bash ; }
dbus-source(){   echo ${BASH_SOURCE:-$(env-home)/$(dbus-src)} ; }
dbus-vi(){       vi $(dbus-source) ; }
dbus-env(){      elocal- ; }
dbus-usage(){ cat << EOU

DBUS : CLI CONTROL OF RUNNING APPS 
====================================

apps using DBUS
---------------

#. **meshlab-**


macports dbus
---------------

I was hoping could just use cmdline to server communication, but it seems this example
needs the dbus server.

* http://trac.macports.org/ticket/20645

Dbus is installed but daemon is not running.

Start macports dbus daemon
----------------------------

::

    simon:~ blyth$ port notes dbus
    dbus has the following notes:
      ############################################################################
      # Startup items have been generated that will aid in
      # starting dbus with launchd. They are disabled
      # by default. Execute the following command to start them,
      # and to cause them to launch at startup:
      #
      # sudo launchctl load -w /Library/LaunchDaemons/org.freedesktop.dbus-system.plist
      # launchctl load -w /Library/LaunchAgents/org.freedesktop.dbus-session.plist
      ############################################################################
::

    simon:meshlab blyth$ launchctl load -w /Library/LaunchAgents/org.freedesktop.dbus-session.plist
    launchctl: CFURLWriteDataAndPropertiesToResource(/Library/LaunchAgents/org.freedesktop.dbus-session.plist) failed: -10
    simon:meshlab blyth$ 

::

    simon:meshlab blyth$ sudo launchctl load -w /Library/LaunchDaemons/org.freedesktop.dbus-system.plist
    simon:meshlab blyth$ ps aux | grep dbus
    messagebus 94546   0.0  0.0    75836    684   ??  Ss    2:05pm   0:00.02 /opt/local/bin/dbus-daemon --system --nofork


Seems the session daemon is started automatically from the system daemon as only one successful load yet 
two daemons, system and session::

    simon:~ blyth$ ps aux | grep dbus
    blyth    94851   0.0  0.0    75848    804   ??  S     2:10pm   0:00.05 /opt/local/bin/dbus-daemon --nofork --session
    messagebus 94546   0.0  0.0    75836    684   ??  Ss    2:05pm   0:00.02 /opt/local/bin/dbus-daemon --system --nofork



Following a reboot no session daemon ?
-----------------------------------------

::

    simon:~ blyth$ ps aux | grep dbus
    messagebus    54   0.0  0.0    75836    172   ??  Ss   Fri06pm   0:00.02 /opt/local/bin/dbus-daemon --system --nofork


Symptom of DBUS server not running
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    simon:bitbucket blyth$ meshlab-v "http://localhost/dae/tree/0.html?c=0.001&a=0,0,-1&fov=60"
    Dynamic session lookup supported but failed: launchd did not provide a socket path, verify that org.freedesktop.dbus-session.plist is loaded!
    Could not connect to D-Bus server: org.freedesktop.DBus.Error.NoMemory: Not enough memory


Some confusion over whether sudo is needed to load it
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* http://trac.macports.org/changeset/56579
* http://trac.macports.org/ticket/20645

Following load no process appears, as on demand.


Also communication still not working
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    simon:~ blyth$ meshlab-v "http://localhost/dae/tree/1.html?c=0.001&a=0,0.5,0"
    Cannot find '.SayHelloThere' in object / at com.meshlab.navigator
    simon:~ blyth$ 

Presumably need to restart the app again after the session daemon is running as
qdbus does not know of the vended API::

    simon:~ blyth$ qdbus 
    :1.2
    org.freedesktop.DBus
    simon:~ blyth$ 

After restarting meshlab::

    simon:~ blyth$ qdbus 
    :1.3
     com.meshlab.navigator
    :1.4
    org.freedesktop.DBus

Now its working.




EOU
}
dbus-dir(){ echo $(local-base)/env/network/network-dbus ; }
dbus-cd(){  cd $(dbus-dir); }
dbus-mate(){ mate $(dbus-dir) ; }
dbus-get(){
   local dir=$(dirname $(dbus-dir)) &&  mkdir -p $dir && cd $dir

}

dbus-session-load(){
   sudo launchctl load -w /Library/LaunchAgents/org.freedesktop.dbus-session.plist
}


