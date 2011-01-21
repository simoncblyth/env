# === func-gen- : base/sv fgp base/sv.bash fgn sv fgh base
sv-src(){      echo base/sv.bash ; }
sv-source(){   echo ${BASH_SOURCE:-$(env-home)/$(sv-src)} ; }
sv-vi(){       vi $(sv-source) ; }
sv-env(){      elocal- ; }
sv-name(){     echo sv-$USER | tr "[A-Z]" "[a-z]" ; }
sv(){          sv-ctl $* ; }
sv-usage(){
  cat << EOU
     sv-src : $(sv-src)
     sv-dir : $(sv-dir)
     sv-confpath : $(sv-confpath)

   == INSTALLATIONS ==

     Production installs mostly done with "easy_install supervisor" into "system" python
     Can determine version with "sv version" when the daemon is running

      C     3.0a7
      C2    3.0a7
      N     3.0a7
      WW    ???      
                    as do not have permissions for system python on IHEP web server 
                    installed into virtual python
      Z     3.0a8

   == UPDATING CONFIG ==

     Config read from the sv-confpath by cfp- 
        sv-logfile : $(sv-logfile)
        sv-pidfile : $(sv-pidfile)
        
            On some nodes these are still stuck in /tmp ... 
            which is a bad location as that is cleaned weekly?
            
            On C,C2  : moved these to /var/log and /var/run to avoid ...        
                 C> maintail
                 supervisord: ERROR (no log file)
    
     To propagate private- changes, use sv-cnf ... you will be shown the diff 
     and will need to confirm it to proceed.  To make other non-private 
     changes simply edit the conf using sv-edit 
     (sv-cnf always says no changes for these as the source is the conf itself). 

     For these to take effect issue a "shutdown" from sv-ctl (or "sv" shortcut)
     and start again using service interface    
          sv-service start 

   == environment parsing bug in 3.0a8 ==
   
       sv-start
       Error: Unexpected end of key/value pairs
       
       Workaround is to quote envirobment values ...
            http://lists.supervisord.org/pipermail/supervisor-users/2010-March/000539.html
       
   == WEB INTERFACE ==

       http://<hostname>:9001/

   == REFERENCES ==    

       http://supervisord.org/manual/current/
       http://pypi.python.org/pypi/superlance/
       http://svn.supervisord.org/superlance/trunk/

    Eventlistener that sends emails on status changes...
       http://lists.supervisord.org/pipermail/supervisor-users/2009-October/000480.html

    A searchable archive ... unlike archaic pipermail :
       http://www.mail-archive.com/supervisor-users@lists.supervisord.org/

    environment parsing issue
       http://www.mail-archive.com/supervisor-users@lists.supervisord.org/msg00345.html

    Bug tracker of sorts ...
       http://www.plope.com/search?SearchableText=supervisord

    Guessed url  
       http://svn.supervisord.org

  == supervisor documentation ==

    Built the supervisor/trunk/docs using sphinx-build at 
       file:///usr/local/env/sv/supervisor/docs/.build/html/configuration.html#fcgi-program-x-section-settings

  == supervisorctl / supervisord experience  ==
    
    * after changing conf must "update", doing "reread" is insufficient 
    * resolve port already bound : 
        * find culprit pid(s) with  {{{sudo lsof -i :4000}}} then use {{{ps aux | grep nnnn}}} to see which programs they are running         
    * avoid messy startup errors from failure to connect to DB by setting webapp priorities later than the db 

  == selection of supervisorctl commands ==

{{{    

avail                   Display all configured processes
add <name> [...]        Activates any updates in config for process/group
remove <name> [...]     Removes process/group from active config
update                  Reload config and add/remove as necessary
reread                  Reload the daemon's configuration files

clear <name>            Clear a process' log files.
clear <name> <name>     Clear multiple process' log files
clear all               Clear all process' log files

fg <process>            Connect to a process in foreground mode Press Ctrl+C to exit foreground

restart <name>          Restart a process
restart <gname>:*       Restart all processes in a group
restart <name> <name>   Restart multiple processes or groups
restart all             Restart all processes

start <name>            Start a process
start <gname>:*         Start all processes in a group
start <name> <name>     Start multiple processes or groups
start all               Start all processes

stop <name>             Stop a process
stop <gname>:*          Stop all processes in a group
stop <name> <name>      Stop multiple processes or groups
stop all                Stop all processes

reload                  Restart the remote supervisord.
shutdown                Shut the remote supervisord down.

}}}


 
  == standard operations ==

    sv-start
                start when using system python 
                on nodes where supervisor has been initd use the {{{/sbin/service}}} interface for consistency :
                   * {{{sudo /sbin/service sv-blyth start}}}

    sv-sstart  
                start when using source python 

                to allow remote access will need to open the port:
                   IPTABLES_PORT=$(sv-ctl-port) iptables-webopen-ip $(local-tag2ip G)
    sv-ps 
                check for the supervisord process

    sv-ctl
               xmlrpc control of remote nodes, used by the shortcuts :
                    sv-C
                    sv-C2
                    sv-N
               assumes common port NNNN between local and remote  as defined in the private-val 

               NB
                If the supervisord is not running on the target (or the port is not open) you will get  : 
                     socket.error: (61, 'Connection refused')
                on attempting to start the controller


  == supervisord control ==

    sv-logfile  : $(sv-logfile)
    sv-tail 
             Follow the tail 

    sv-pidfile  : $(sv-pidfile)
    sv-pid
             pid of the supervisord 
             NB keeping the pidfile in /tmp is not wise for such a longlived process 
             ... better to keep in /var/run 
    
    sv-restart 
              send HUP for supervisord which restarts it .. re-reading config

    sv-stop 
              stops the supervisor either via the /sbin/service or via sv-ctl depending 
              on the node, in the latter case you will need to manually confirm for the 
              shutdown to proceed 


  == making supervisord auto start on reboot ==
     
     NB stopping/starting  supervisord will stop/start  all the processes it controls

     sv-initd-path : $(sv-initd-path)
     sv-initd  
         set up the ini script for supervisor
     sv-initd-
         emit the script to stdout 

     To test when using source python as on C2 : need to do sudo bash python hoop jumping : 
          [blyth@cms02 e]$ python-
          [blyth@cms02 e]$ sudo bash -c "LD_LIBRARY_PATH=$(python-libdir) /etc/rc.d/init.d/sv-blyth start "


     To bring it into the chkconfig fold...

      [blyth@cms01 ~]$ sudo chkconfig --add sv-blyth
      [blyth@cms01 ~]$ sudo chkconfig --list sv-blyth
      sv-blyth        0:off   1:off   2:on    3:on    4:on    5:on    6:off
      [blyth@cms01 ~]$ sudo service sv-blyth status


      [blyth@cms01 ~]$ sudo service sv-blyth start
      Starting sv-blyth: /data/env/system/python/Python-2.5.1/bin/python: error while loading shared libraries: libpython2.5.so.1.0: cannot open shared object file: No such file or directory


  == ini issue ... running on the wrong python == 

    * resolved using \$(which python) in the svconf ... as init.d environment is bare ... unlike manual start with : sv-sstart

  == putting processes under supervisor control ==

    sv-plus ininame
        Replacement for sv-add 
           rather than passing the name of the function that emits the config 
           ... simply pipe the config in allow to pass arguments to the func 
           ... avoiding straightjacket 

        Usage :
             whatever arg1 arg2 | sv-plus whatever.ini

     sv-add fnc ininame   (DEPRECATED ... NOW USING sv-plus)

        Adds the config to sv-confdir:$(sv-confdir), 
        for inclusion at the next supervisor reload.  

                fnc : name of function that emits supervisor config to stdout
            ininame : name of the supervisor config file, 
                      eg: hgweb.ini, runinfo.ini, plvdbi.ini  

        NB will also need to hookup up the mount point in apache/lighttpd/nginx 
           for SCGI/FCGI webapps 



   == getting / building / initial config ==


     sv-get
         superlance, required a setuptools update 
              "The required version of setuptools (>=0.6c9) is not available, and
               a more recent version first, using 'easy_install -U setuptools'."

         it makes most sense to install supervisor into the base python rather than
         virtual pythons 

     sv-bootstrap
        initial setup , based on private config values

     sv-cnf
          apply the sv-cnf-triplets- to the supervisord config 
          (operates via sv-ini)

     sv-cnf-triplets-
          supervisord config edit triplets used by sv-cnf 

          inet is essential ... so remove unix_http_server for simplicity   
          avoiding :  
                2009-10-06 16:07:50,845 CRIT Server 'unix_http_server' running without any HTTP authentication checking


   == ALTERNATIVES ==

       * http://www.puppetlabs.com/company/overview/
  
       * http://www.agileweboperations.com/monitoring-tools-essentials-munin-vs-nagios/

       * munin http://munin-monitoring.org/
          * perlbased server-node system built on RRDtool (round robin database .. designed for time series data )
             * http://oss.oetiker.ch/rrdtool/ 
             * RRD problem ... : inflexibility wrt the '''step'''
             * munin default step 5 min .. some added flexibility in 1.6? (not yet in distro versions) 
        * easy to write plugins (simple bash scripts even)
             * many plugins available   


       * nagios 
          * http://www.nagios.org/
          * http://exchange.nagios.org/
          * http://docs.pnp4nagios.org/pnp-0.4/start   RRD 

       * monitoring rabbitmq with nagios
          * http://morethanseven.net/2010/01/30/rabbitmq-support-cucumber-nagios.html 

       * http://serverfault.com/questions/97270/munin-vs-nagios

Munin and Nagios are really different tools.
from munin website:
    Munin is a networked resource monitoring tool that can help analyze resource trends and "what just happened to kill our performance?" problems. It is designed to be very plug and play. A default installation provides a lot of graphs with almost no work.

Nagios is a monitoring (alerting) tool. Munin could be a replacenment of Cacti
we use both of them: Nagios and Munin. Nagios tell us in real time if something is wrong: like web server down, db load average, etc.

Using munin you can see the trends and the history about why that happenend.
link|flag

answered Dec 24 '09 at 21:06
Gabriel Sosa
29027


EOU
}
sv-dir(){      echo $(local-base)/env/sv ; }
sv-confpath(){ echo $(sv-dir)/supervisord.conf ; }
sv-confdir(){  echo $(sv-dir)/conf ; }
sv-ctldir(){   echo $(sv-dir)/ctl  ; }
sv-cd(){  cd $(sv-dir); }
sv-mate(){ mate $(sv-dir) ; }

sv-baseurl(){  echo http://svn.supervisord.org ; }


sv-get-trunk(){
   local dir=$(sv-dir) && mkdir -p $dir && cd $dir
   svn co $(sv-baseurl)/supervisor/trunk supervisor
   svn co $(sv-baseurl)/superlance/trunk superlance
}

sv-trunk-html(){
   open $(sv-dir)/supervisor/docs/.build/html/index.html
}


sv-get(){
   python-
   local v=$(python-v)
   case $(which easy_install) in
     /usr/bin/easy_install*)  sudo=sudo ;;
                          *)  sudo=""   ;;
   esac
   $sudo easy_install$v supervisor
   $sudo easy_install$v superlance    ## memmon + httpok 
}

sv-build(){
   sv-get
   sv-bootstrap
}

sv-bootstrap(){
   local msg="=== $FUNCNAME :"
   [ "$(which echo_supervisord_conf)" == "" ] && echo $msg ABORT supervisor not installed in this python && return 1
   local conf=$(sv-confpath)
   mkdir -p $(dirname $conf) 
   echo $msg writing to $conf
   echo_supervisord_conf  > $conf.sample   ## will be loosing the comments so write a sample to preserve them
   echo_supervisord_conf  > $conf
   mkdir -p $(sv-confdir)
   sv-cnf
}


sv-bins(){ echo supervisorctl supervisord echo_supervisord_conf pidproxy ; }
sv-check(){
   local msg="=== $FUNCNAME :"
   local bin ; for bin in $(sv-bins) ; do
     echo $msg $bin $(which $bin)
   done
}
sv-edit(){ vim $(sv-confpath) $(sv-confdir)/*.ini $(sv-confpath).sample ; }
sv-sample(){ vim $(sv-confpath).sample ; }

sv-user(){   echo ${SV_USER:-root} ; }
sv-sudo(){   
    case $(sv-user) in
        root) echo sudo ;;
       $USER) echo -n   ;;
           *) echo sudo ;;
    esac
 }


sv-start(){
   case $(hostname) in 
      cms01.phys.ntu.edu.tw) type $FUNCNAME && sudo /sbin/service sv-blyth start ;;
                          *)  $FUNCNAME- ;;
   esac
}

sv-stop(){
   case $(hostname) in 
      cms01.phys.ntu.edu.tw) type $FUNCNAME && sudo /sbin/service sv-blyth stop ;;
                          *)  sv-ctl shutdown ;;  ## will be prompted for confirmation
   esac
}





sv-start-(){  $(sv-sudo) supervisord   -c $(sv-confpath)    $* ; } 
sv-nstart(){ $(sv-sudo) supervisord   -c $(sv-confpath) -n $* ; }   ## -n ... non daemon useful for debugging 
sv-ps(){     ps aux | grep -v grep | grep supervisord  ; }

sv-service(){
  local arg=${1:-start}
  local msg="=== $FUNCNAME :"
  local cmd="sudo /sbin/service $(sv-name) $arg "
  echo $msg $cmd
  eval $cmd
}


sv-sstart(){
   ## when using source python, have to jump thu "sudo python" hoops ...  
   python-
   sudo bash -c "LD_LIBRARY_PATH=$(python-libdir) supervisord -c $(sv-confpath)" 
}


sv-add(){
   local msg="=== $FUNCNAME :"
   echo $msg THIS IS DEPRECATED USE sv-plus && return 1
   local fnc=$1
   local nam=$2
   local tmp=/tmp/env/$FUNCNAME/$nam && mkdir -p $(dirname $tmp)
   $fnc
   $fnc > $tmp
   local cmd="sudo cp $tmp $(sv-confdir)/ "
   echo $msg $cmd
   eval $cmd
}

sv-plus(){
  local msg="=== $FUNCNAME :"
  local nam=$1
  local tmp=/tmp/$USER/env/$FUNCNAME/$nam && mkdir -p $(dirname $tmp)
  echo $msg writing to $tmp
  cat - > $tmp
  cat $tmp
  local ini=$(sv-confdir)/$nam 
  echo $msg proposed change to $ini ...
  if [ -f "$ini" ]; then
     diff $ini $tmp
  fi 
  local cmd="sudo cp $tmp $ini "
  
  # taking input and receiving from a pipe ... dont work well together 
  #local ans
  #read -p "$msg Enter YES to proceed with : $cmd" ans
  #[ ! "$ans" == "YES" ] && echo $msg skipping .. && return 0
  #echo $msg proceeding ... 
  
  eval $cmd
}


sv-ini(){ 
  ini-
  INI_TRIPLET_DELIM="|" INI_FLAVOR="ini_cp" ini-triplet-edit $(sv-confpath) $*  
}

sv-sha-(){ python -c "import hashlib ; print \"{SHA}%s\" % hashlib.sha1(\"${1:-thepassword}\").hexdigest() " ; } 
sv-sha(){
  local pass=$1
  if [ "${pass:0:5}" == "{SHA}" ]; then
    echo $pass
  else
    sv-sha- $pass 
  fi 
}

sv-cnf-triplets-(){   
  private-
  cat << EOC

  include|files|conf/*.ini

  inet_http_server|port|$(private-val SUPERVISOR_PORT) 
  inet_http_server|username|$(private-val SUPERVISOR_USERNAME)
  inet_http_server|password|$(sv-sha $(private-val SUPERVISOR_PASSWORD))


  supervisord|user|$(sv-user) 
  supervisord|logfile|$(sv-logfile)
  supervisord|pidfile|$(sv-pidfile)

  supervisorctl||
  unix_http_server||

EOC
}

sv-tail(){    tail -f $(sv-logfile) ; }
sv-logfile(){ cfp- ; cfp-getset supervisord logfile ; }
sv-pidfile(){ cfp- ; cfp-getset supervisord pidfile ; }
sv-pid(){ 
   local pidfile=$(sv-pidfile)
   if [ -f "$pidfile" ]; then
      cat $(sv-pidfile) 
   else
      sv-ctl pid
   fi
}

sv-restart(){ 
   local msg="=== $FUNCNAME :"
   local cmd="$(sv-sudo) kill -HUP $(sv-pid)" 
   echo $msg $cmd
   eval $cmd  
}

sv-ctl-sock(){
   local user=$(private-val SUPERVISOR_USERNAME)
   local sock=/tmp/env/$user/supervisor.sock
   echo $sock
}

sv-cnf-triplets-nonet-(){
   private-
   local sock=$(sv-ctl-sock)
   mkdir -p $(dirname $sock)
   cat << EOC

  include|files|conf/*.ini

  unix_http_server|file|$(sv-ctl-sock)
  unix_http_server|chmod|0777
  unix_http_server|chown|$(sv-user):$(sv-user)
  unix_http_server|username|$(private-val SUPERVISOR_USERNAME)
  unix_http_server|password|$(private-val SUPERVISOR_PASSWORD)

  supervisord|user|$(sv-user) 

  supervisorctl||
  inet_http_server||

EOC

# must leave this section in there .... rpcinterface:supervisor||

}



sv-private-check(){

  local msg="=== $FUNCNAME :"
  private-
  local port=$(private-val SUPERVISOR_PORT)
  local user=$(private-val SUPERVISOR_USERNAME)
  local pass=$(private-val SUPERVISOR_PASSWORD)

  if [ -z "$port" -o -z "$user" -o -z "$pass" ]; then
       echo $msg ABORT missing private-val && type $FUNCNAME && return 1
  fi 
}

sv-cnf(){ 
   sv-private-check
   [ ! "$?" -eq  "0" ] && echo ABORTED && return $?
   sv-ini $(sv-cnf-triplets-);  
}

sv-cnf-nonet(){ 
   sv-private-check
   [ ! "$?" -eq  "0" ] && echo ABORTED && return $?
   sv-ini $(sv-cnf-triplets-nonet-);  
}





##  supervisorctl config for controlling a network of nodes over xmlrpc

sv-G(){  SV_TAG=G  sv-ctl $* ; }
sv-C(){  SV_TAG=C  sv-ctl $* ; }
sv-N(){  SV_TAG=N  sv-ctl $* ; }
sv-H(){  SV_TAG=H  sv-ctl $* ; }
sv-C2(){ SV_TAG=C2 sv-ctl $* ; }
sv-WW(){ SV_TAG=WW sv-ctl $* ; }


sv-supervisorctl(){
   case $(hostname -s) in 
      cms01) echo /data/env/system/python/Python-2.5.1/bin/supervisorctl ;;
          *) echo supervisorctl ;;
   esac
}


sv-ctl(){ 
   local msg="=== $FUNCNAME :"
   local ini=$(sv-ctl-ini)
   #[ "$(which supervisorctl)" == "" ] && echo $msg no supervisorctl ... you are running with the wrong python && return 1
   [ ! -f "$ini" ] && echo $msg ABORT no ini $ini for tag $(sv-ctl-tag) ... use \"SV_TAG=$(sv-ctl-tag) sv-ctl-prep\" to create one  && return 1
   local cmd="$(sv-supervisorctl) -c $ini $*  "
   [ -n "$SV_VERBOSE" ] && echo $msg $cmd
   eval $cmd
   [ ! "$?" -eq "0" ] && echo $msg error maybe server not running ... the ini $ini : && cat $ini
}
sv-ctl-tag(){ echo ${SV_TAG:-$NODE_TAG} ; }
sv-ctl-ini(){ echo $(sv-ctldir)/$(sv-ctl-tag).ini ; }
sv-ctl-port(){
  private-
  local hostport=$(private-val SUPERVISOR_PORT)
  local port=${hostport:${#hostport}-4}           ## last 4 chars give the port number
  echo $port
}
sv-ctl-prep-(){
  private-
  local tag=$(sv-ctl-tag)
  local port=$(sv-ctl-port)
  local ip
  if [ "$tag" == "$NODE_TAG" ]; then
      ip=127.0.0.1
  else
      ip=$(local-tag2ip $tag)
  fi 

  local username=$(private-val SUPERVISOR_USERNAME_$tag)
  [ -z "$username" ] && username=$(private-val SUPERVISOR_USERNAME)

  local password=$(private-val SUPERVISOR_PASSWORD_$tag)
  [ -z "$password" ] && password=$(private-val SUPERVISOR_PASSWORD)



  local server=$ip:$port   
  cat << EOC
[supervisorctl]
serverurl=http://$server 
username=$username
password=$password
prompt=$tag 
history_file=~/.svctl.$tag  
EOC
}
sv-ctl-prep(){
   local msg="=== $FUNCNAME :"
   local ini=$(sv-ctl-ini)
   local dir=$(dirname $ini) && mkdir -p $dir
   echo $msg creating $ini
   $FUNCNAME- $* > $ini
   chmod go-rwx $ini
   cat $ini
}


sv-ctl-prep-nonet-(){
   cat << EOC
[supervisorctl]
serverurl = unix://$(sv-ctl-sock)
username = $(private-val SUPERVISOR_USERNAME)
password = $(private-val SUPERVISOR_PASSWORD)
prompt = nonet

EOC
}

sv-ctl-prep-nonet(){
   local msg="=== $FUNCNAME :"
   local ini=$(sv-ctl-ini)
   local dir=$(dirname $ini) && mkdir -p $dir
   echo $msg creating $ini
   $FUNCNAME- $* > $ini
   chmod go-rwx $ini
   cat $ini
}



sv-webopen-tag(){
   local tag=${1:-G}
   local ip=$(local-tag2ip $tag)  
   sv-webopen-ip $ip 
}

sv-webopen-ip(){
   local ip=$1
   local port=$(sv-ctl-port)
   iptables-
   IPTABLES_PORT=$port iptables-webopen-ip $ip
}



sv-dev-url(){ echo http://svn.supervisord.org/supervisor/trunk/ ; }
sv-dev-dir(){ echo $(local-base)/env/supervisor-dev ; }
sv-dev-cd(){ cd $(sv-dev-dir) ; }
sv-dev-get(){
  local dir=$(sv-dev-dir)
  mkdir -p $dir 
  sv-dev-cd
  [ ! -d trunk ] && svn co $(sv-dev-url) 
 
}

sv-dev-install(){
  sv-dev-cd
  cd trunk
  python setup.py install
}


sv-httpok-conf-(){ private- ; cat << EOC
[eventlistener:httpok_plvdbi]
command=$(which python) -u $(which httpok) -p plvdbi -m $(local-email) http://dayabay.phys.ntu.edu.tw/dbi/
events=TICK_3600
redirect_stderr=true
redirect_stdout=true
environment=SUPERVISOR_USERNAME=$(private-val SUPERVISOR_USERNAME),SUPERVISOR_PASSWORD=$(private-val SUPERVISOR_PASSWORD),SUPERVISOR_SERVER_URL=$(private-val SUPERVISOR_SERVER_URL)
EOC
}

sv-httpok-conf(){
   local msg="=== $FUNCNAME :"
   local ini=$(sv-confdir)/httpok.ini
   echo $msg writing $ini
   $FUNCNAME- > $ini
   chmod go-rw $ini
   ll $ini
   cat $ini
}

sv-initd-(){ python- ; cat << EOI
#!/bin/sh
# RedHat startup script for a supervisor instance
#    http://lists.supervisord.org/pipermail/supervisor-users/2007-September/000093.html
#
# chkconfig: 2345 80 20
# description: $(sv-name)
#
# Source function library.
. /etc/rc.d/init.d/functions


export LD_LIBRARY_PATH=$(python-libdir):\$LD_LIBRARY_PATH
export PATH=$(python-bindir):\$PATH

py=\$(which python)
[ "\$py" != "/usr/bin/python" ] && echo using py \$py  

pv=\$(python -V 2>&1)
[ "\$pv" != "Python 2.5.1" ] && echo wrong python \$pv  && exit 1

name="$(sv-name)"

RETVAL=0

start() {
     echo -n "Starting \$name: "
     supervisord -c $(sv-confpath)
     RETVAL=\$?
     [ \$RETVAL -eq 0 ] && touch /var/lock/subsys/\$name
     echo
     return \$RETVAL
}

stop() {
     echo -n "Stopping \$name: "
     supervisorctl -c $(sv-ctl-ini) shutdown
     RETVAL=\$?
     [ \$RETVAL -eq 0 ] && rm -f /var/lock/subsys/\$name
     echo
     return \$RETVAL
}

case "\$1" in
        start) start ;;
         stop) stop  ;;
      restart)
               stop
               start
                     ;;
esac

exit \$REVAL
EOI
}

sv-initd-path(){ echo /etc/init.d/$(sv-name) ; }
sv-initd(){
  local msg="=== $FUNCNAME :"
  local ini=$(sv-initd-path)
  [ ! -d "$(dirname $ini)" ] && echo $msg error no init.d for $ini && return 1
  local tmp=/tmp/$USER/env/$FUNCNAME/$(sv-name) && mkdir -p $(dirname $tmp)
  $FUNCNAME- > $tmp
  local cmd
  if [ -f "$ini" ]; then 
     cmd="diff $ini $tmp"
     echo $msg $cmd
     eval $cmd
  fi
  cmd="sudo cp $tmp $ini "
  local ans
  read -p "$msg enter YES to proceed with : $cmd " ans
  [ "$ans" != "YES" ] && echo $msg OK skipping && return 0
  eval $cmd 

  cmd="sudo chmod ugo+x $ini "
  eval $cmd
  ls -l $ini

}


sv-dumpenv-(){ 
  private-
cat << EOD
[program:dumpenv]
command=env
environment=ENV_PRIVATE_PATH='$(private-path)',ENV_HOME='$ENV_HOME'
EOD
}

sv-dumpenv(){
   $FUNCNAME- | sv-plus dumpenv.ini   
}

