# === func-gen- : base/sv fgp base/sv.bash fgn sv fgh base
sv-src(){      echo base/sv.bash ; }
sv-source(){   echo ${BASH_SOURCE:-$(env-home)/$(sv-src)} ; }
sv-vi(){       vi $(sv-source) ; }
sv-env(){      elocal- ; }
sv-usage(){
  cat << EOU
     sv-src : $(sv-src)
     sv-dir : $(sv-dir)
     sv-confpath : $(sv-confpath)

       http://supervisord.org/manual/current/
     
  == standard operations ==

    sv-start
                start when using system python 

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





EOU
}
sv-dir(){      echo $(local-base)/env/sv ; }
sv-confpath(){ echo $(sv-dir)/supervisord.conf ; }
sv-confdir(){  echo $(sv-dir)/conf ; }
sv-ctldir(){   echo $(sv-dir)/ctl  ; }
sv-cd(){  cd $(sv-dir); }
sv-mate(){ mate $(sv-dir) ; }
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
sv-edit(){ vim $(sv-confpath) $(sv-confdir)/*.ini ; }
sv-sample(){ vim $(sv-confpath).sample ; }

sv-user(){   echo ${SV_USER:-root} ; }
sv-sudo(){   
    case $(sv-user) in
        root) echo sudo ;;
       $USER) echo -n   ;;
           *) echo sudo ;;
    esac
 }
sv-start(){  $(sv-sudo) supervisord   -c $(sv-confpath)    $* ; } 
sv-nstart(){ $(sv-sudo) supervisord   -c $(sv-confpath) -n $* ; }   ## -n ... non daemon useful for debugging 
sv-ps(){     ps aux | grep -v grep | grep supervisord  ; }


sv-sstart(){
   ## when using source python, have to jump thu "sudo python" hoops ...  
   python-
   sudo bash -c "LD_LIBRARY_PATH=$(python-libdir) supervisord -c $(sv-confpath)" 
}


sv-add(){
   local msg="=== $FUNCNAME :"
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
  local tmp=/tmp/env/$FUNCNAME/$nam && mkdir -p $(dirname $tmp)
  echo $msg writing to $tmp
  cat - > $tmp
  cat $tmp
  local cmd="sudo cp $tmp $(sv-confdir)/ "
  echo $msg $cmd
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

  supervisorctl||
  unix_http_server||

EOC
}


sv-pid(){ cat /tmp/supervisord.pid ; }
sv-restart(){ kill -HUP $(sv-pid) ;  }

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

sv-ctl(){ 
   local msg="=== $FUNCNAME :"
   local ini=$(sv-ctl-ini)
   [ "$(which supervisorctl)" == "" ] && echo $msg no supervisorctl ... you are running with the wrong python && return 1
   [ ! -f "$ini" ] && echo $msg ABORT no ini $ini for tag $(sv-ctl-tag) ... use \"SV_TAG=$(sv-ctl-tag) sv-ctl-prep\" to create one  && return 1
   local cmd="supervisorctl -c $ini $*  "
   echo $msg $cmd
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
  local server=$ip:$port   
  cat << EOC
[supervisorctl]
serverurl=http://$server 
username=$(private-val SUPERVISOR_USERNAME) 
password=$(private-val SUPERVISOR_PASSWORD)
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





sv-webopen-ip(){
   local tag=${1:-G}
   local port=$(sv-ctl-port)
   iptables-
   IPTABLES_PORT=$port iptables-webopen-ip $(local-tag2ip $tag)  
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




