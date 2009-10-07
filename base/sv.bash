# === func-gen- : base/sv fgp base/sv.bash fgn sv fgh base
sv-src(){      echo base/sv.bash ; }
sv-source(){   echo ${BASH_SOURCE:-$(env-home)/$(sv-src)} ; }
sv-vi(){       vi $(sv-source) ; }
sv-env(){      elocal- ; }
sv-usage(){
  cat << EOU
     sv-src : $(sv-src)
     sv-dir : $(sv-dir)

       http://supervisord.org/manual/current/

     sv-get
         superlance, required a setuptools update 
              "The required version of setuptools (>=0.6c9) is not available, and
               a more recent version first, using 'easy_install -U setuptools'."


     sv-confpath : $(sv-confpath)

     sv-add fnc ininame

        Adds the config to sv-confdir:$(sv-confdir), 
        for inclusion at the next supervisor reload.  

                fnc : name of function that emits supervisor config to stdout
            ininame : name of the supervisor config file, 
                      eg: hgweb.ini, runinfo.ini, plvdbi.ini  

        NB will also need to hookup up the mount point in apache/lighttpd/nginx 
           for SCGI/FCGI webapps 

     sv-cnf
          apply the sv-cnf-triplets- to the supervisord config 
          (operates via sv-ini)

     sv-cnf-triplets-
          supervisord config edit triplets used by sv-cnf 

          inet is essential ... so remove unix_http_server for simplicity   
          avoiding :  
                2009-10-06 16:07:50,845 CRIT Server 'unix_http_server' running without any HTTP authentication checking

     sv-ctl
         xmlrpc control of remote nodes  
         assumes common port NNNN between local and remote  as defined in the private-val 



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
sv-edit(){ vim $(sv-confpath) ; }
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


sv-ini(){ sv-ini- $(sv-confpath) $*  ; }
sv-ini-() 
{ 
    local msg="=== $FUNCNAME :";
    local path=$1 ;
    shift;
    local tmp=/tmp/env/$FUNCNAME && mkdir -p $tmp;
    local tpath=$tmp/$(basename $path);
    local cmd="cp $path $tpath ";
    eval $cmd;
    
    ## NB using a ConfigParser variant of the ConfigObj original as supervisor ini files
    ## use inline comments and semicolons that give ConfigParser indigestion
    
    INI_TRIPLET_DELIM="|" python $ENV_HOME/base/ini_cp.py $tpath $*;
    local dmd="diff $path $tpath";
    echo $msg $dmd;
    eval $dmd;
    [ "$?" == "0" ] && echo $msg no differences ... skipping && return 0;
    if [ -n "$SV_CONFIRM" ]; then
        local ans;
        read -p "$msg enter YES to confirm this change " ans;
        [ "$ans" != "YES" ] && echo $msg skipped && return 1;
    fi;
    $SUDO cp $tpath $path;


    #[ "$user" != "$USER" ] && $SUDO chown $user:$user $path
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



##  supervisorctl config for controlling a network of nodes over xmlrpc

sv-C(){  SV_TAG=C  sv-ctl $* ; }
sv-N(){  SV_TAG=N  sv-ctl $* ; }
sv-H(){  SV_TAG=H  sv-ctl $* ; }
sv-C2(){ SV_TAG=C2 sv-ctl $* ; }

sv-ctl(){ 
   local msg="=== $FUNCNAME :"
   local ini=$(sv-ctl-ini)
   [ ! -f "$ini" ] && echo $msg ABORT no ini $ini for tag $(sv-ctl-tag) ... use \"SV_TAG=$(sv-ctl-tag) sv-ctl-prep\" to create one  && return 1
   local cmd="supervisorctl -c $ini $*  "
   echo $msg $cmd
   eval $cmd
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
  local remote=$(local-tag2ip $tag):$port   
  cat << EOC
[supervisorctl]
serverurl=http://$remote 
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
}

sv-webopen-ip(){
   local tag=${1:-G}
   local port=$(sv-ctl-port)
   iptables-
   IPTABLES_PORT=$port iptables-webopen-ip $(local-tag2ip $tag)  
}
