# === func-gen- : messaging/rabbitmq fgp messaging/rabbitmq.bash fgn rabbitmq fgh messaging
rabbitmq-src(){      echo messaging/rabbitmq.bash ; }
rabbitmq-source(){   echo ${BASH_SOURCE:-$(env-home)/$(rabbitmq-src)} ; }
rabbitmq-vi(){       vi $(rabbitmq-source) ; }
rabbitmq-env(){      elocal- ; }

rabbitmq-actionlog(){ cat << EOL



  17/02/2011 ... G  (with the port py26) ... rabbitmq 2.3.1 
            sudo port selfupdate
            sudo port install rabbitmq-server

simon:~ blyth$ sudo port install rabbitmq-server
Portfile changed since last build; discarding previous state.
--->  Computing dependencies for libxml2
--->  Fetching libxml2
--->  Attempting to fetch libxml2-2.7.8.tar.gz from http://distfiles.macports.org/libxml2
--->  Verifying checksum(s) for libxml2
--->  Extracting libxml2
--->  Configuring libxml2
--->  Building libxml2
--->  Staging libxml2 into destroot
--->  Computing dependencies for libxml2
--->  Installing libxml2 @2.7.8_0
--->  Deactivating libxml2 @2.7.7_0
--->  Activating libxml2 @2.7.8_0
--->  Cleaning libxml2
--->  Computing dependencies for rabbitmq-server
--->  Dependencies to be installed: erlang perl5 py26-simplejson
--->  Fetching perl5
--->  Verifying checksum(s) for perl5
--->  Extracting perl5
--->  Configuring perl5
--->  Building perl5
--->  Staging perl5 into destroot
--->  Installing perl5 @5.8.9_0
--->  Activating perl5 @5.8.9_0
--->  Cleaning perl5
--->  Fetching erlang
--->  Attempting to fetch otp_src_R14B01.tar.gz from http://distfiles.macports.org/erlang
--->  Attempting to fetch otp_doc_man_R14B01.tar.gz from http://distfiles.macports.org/erlang
--->  Attempting to fetch otp_doc_html_R14B01.tar.gz from http://distfiles.macports.org/erlang
--->  Verifying checksum(s) for erlang
--->  Extracting erlang
--->  Applying patches to erlang
--->  Configuring erlang                     
--->  Building erlang
--->  Staging erlang into destroot
--->  Installing erlang @R14B01_1
--->  Activating erlang @R14B01_1
--->  Cleaning erlang
--->  Fetching py26-simplejson
--->  Attempting to fetch simplejson-2.1.2.tar.gz from http://distfiles.macports.org/python
--->  Verifying checksum(s) for py26-simplejson
--->  Extracting py26-simplejson
--->  Configuring py26-simplejson
--->  Building py26-simplejson
--->  Staging py26-simplejson into destroot
--->  Installing py26-simplejson @2.1.2_0
--->  Activating py26-simplejson @2.1.2_0
--->  Cleaning py26-simplejson
--->  Fetching rabbitmq-server
--->  Attempting to fetch rabbitmq-server-2.3.1.tar.gz from http://distfiles.macports.org/rabbitmq-server
--->  Attempting to fetch rabbitmq-server-generic-unix-2.3.1.tar.gz from http://distfiles.macports.org/rabbitmq-server
--->  Verifying checksum(s) for rabbitmq-server
--->  Extracting rabbitmq-server
--->  Configuring rabbitmq-server
--->  Building rabbitmq-server
--->  Staging rabbitmq-server into destroot
--->  Creating launchd control script
###########################################################
# A startup item has been generated that will aid in
# starting rabbitmq-server with launchd. It is disabled
# by default. Execute the following command to start it,
# and to cause it to launch at startup:
#
# sudo port load rabbitmq-server
###########################################################
--->  Installing rabbitmq-server @2.3.1_0
--->  Activating rabbitmq-server @2.3.1_0
--->  Cleaning rabbitmq-server


 


  17/02/2011 ... N : bringing rabbitmq up to 2.2 

         [blyth@belle7 ~]$ sudo yum --enablerepo=epel upgrade rabbitmq-server


  15/06/2010  
       on N ... use the below to do rougly the same as on C
 
            rabbitmq-register <_1/2/3/...>   .... add users and set permissions 
            rabbitmq-setup                   .... delete guest user
            rabbitmq-wideopen

         check connection with {{{bunny-;bunny-- OTHER_}}}


  13/02/2010 : 
       on C server add_user "a" and change password of "guest" using :
             sudo rabbitmqctl ...
            give a the same permissions as guest  
              
        sudo rabbitmqctl set_permissions a ".*" ".*" ".*"
          > Setting permissions for user "a" in vhost "/" ...

        sudo rabbitmqctl list_permissions
          > Listing permissions in vhost "/" ...
          > a       .*      .*      .*
          > guest   .*      .*      .*

       open the cms01 port for general access
             rabbitmq-wideopen 

EOL
}

rabbitmq-usage(){
  cat << EOU
     rabbitmq-src : $(rabbitmq-src)
     rabbitmq-dir : $(rabbitmq-dir)


  == monitoring tools to try ==

      * http://www.lshift.net/blog/2009/11/30/introducing-rabbitmq-status-plugin
      * http://www.rabbitmq.com/faq.html#topic-exchange

  == rabbitmq server control ==

     rabbitmq-status
     rabbitmq-start
     rabbitmq-stop
           ini controls of the rabbitmq node

  == rabbitmqctl status calls ==

     rabbitmq-exchanges
     rabbitmq-queues
     rabbitmq-bindings
     rabbitmq-connections
               interrogate the server to provide listings of diagnostic fields, 
               provide fieldname arguments to restrict to a subset of the fields 
 
  == man pages ==

   man rabbitmq-server
   man rabbitmq.conf
   man rabbitmq-multi
   man rabbitmqctl

   == installing rabbitmq-server on redhat ==

      I have used the redhat EPEL distrib to get the server with yum
      avoiding hassles of building from source 

   === iptables setting to allow access to the amqp port === 

       rabbitmq-open-ip 140.112.XXX.XX 

   === automating rabbitmq-server launch on reboot ===

     Controlled using chkconfig scripts as rabbitmq-server comes fully redhat integrated from EPEL..
        http://wiki.linuxquestions.org/wiki/Run_Levels

   === service interface ===

    [blyth@cms01 e]$ sudo service rabbitmq-server start
    Starting rabbitmq-server: SUCCESS
    rabbitmq-server.


   == installing rabbitmq-server on OSX : did not pursue due to excessive dependencies ==
 
      * http://trac.macports.org/browser/trunk/dports/net/rabbitmq-server/Portfile
      * http://trac.macports.org/browser/trunk/dports/lang/erlang/Portfile
         * Why does erlang depend on wxWidgets ?
         *  http://www.erlang.org

   == umbrella etc.. ==

     * needed when using rabbitmq plugins .. like Alice ? 
         * http://www.rabbitmq.com/plugin-development.html#getting-started 

     rabbitmq-umbrella-get
     rabbitmq-umbrella-make
            umbrella is simply just a Makefile, that can checkout all rabbitmq-* and build

  == rabbitmq-c ==

      rabbitmq-c building is split into rmqc-


  == rabbitmq examples based on python amqplib ==

     rabbitmq-ex-consumer
     rabbitmq-ex-publisher
            try py-ampqlib based consumer and publisher


  == bunny ==

    an interactive client that provides a great way to
    learn how to hook up exchanges/queues etc..
       bunny-;bunny-usage

    see wiki:RabbitMQFanout


    also can do simple admin : deleting / purging etc 



  == pika ==

    http://www.slideshare.net/hungryblank/rabbitmq-with-python-and-ruby-rupy-2009


  == rabbitmq plugins of interest ==

     last value cache exchange 
         https://github.com/squaremo/rabbitmq-lvc-plugin

  == about rabbitmq ==

     talk by core rabbitmq dev 
         http://www.erlang-factory.com/conference/SFBay2010/speakers/TonyGarnockJones
         http://homepages.kcbbs.gen.nz/tonyg/talks.html

EOU
}
rabbitmq-dir(){ echo $(local-base)/env/messaging ; }
rabbitmq-cd(){  cd $(rabbitmq-dir); }
rabbitmq-mate(){ mate $(rabbitmq-dir) ; }

rabbitmq-logdir(){   echo /var/log/rabbitmq ; }
rabbitmq-logpath(){  echo $(rabbitmq-logdir)/rabbit.log ; }
rabbitmq-log(){    sudo vi $(rabbitmq-logpath) ; }
rabbitmq-tail(){   sudo tail -f $(rabbitmq-logpath) ; }
rabbitmq-confpath(){ echo /etc/rabbitmq/rabbitmq.conf ; }
rabbitmq-edit(){     sudo vi $(rabbitmq-confpath) ; }
rabbitmq-cookie(){   echo /var/lib/rabbitmq/.erlang.cookie ; }

rabbitmq-top(){      top -U rabbitmq $* ; }

rabbitmq-inipath(){ echo /etc/init.d/rabbitmq-server ; }
rabbitmq-ini(){     sudo $(rabbitmq-inipath) $* ; }

rabbitmq-chkconfig(){
   local msg="=== $FUNCNAME :"
   local cmd="sudo chkconfig --levels 345 rabbitmq-server on"
   echo $msg $cmd
   eval $cmd

   cmd="sudo chkconfig --list rabbitmq-server" 
   eval $cmd
}

rabbitmq-ctl(){     sudo rabbitmqctl $* ; }



rabbitmq-user(){ echo $(private- ; private-val RABBITMQ_USER_${1:-0}) ; }
rabbitmq-pass(){ echo $(private- ; private-val RABBITMQ_PASS_${1:-0}) ; }

rabbitmq-register-(){
  local msg="=== $FUNCNAME :"
  local id="${1:-0}" 

  local user=$(rabbitmq-user $id)
  [ -z "$user" ] && echo $msg no user for id $id && return 0

  rabbitmq-ctl add_user $user $(rabbitmq-pass $id)
  ## do directly as tricky to pass such params 
  sudo rabbitmqctl set_permissions $user '.*' '.*' '.*'
  rabbitmq-ctl list_permissions 
}


rabbitmq-start(){   rabbitmq-ini start ; }
rabbitmq-status(){  rabbitmq-ini status ; }
rabbitmq-stop(){    rabbitmq-ini stop ; }


rabbitmq-reset(){

  rabbitmq-ctl stop_app
  rabbitmq-ctl reset
  rabbitmq-ctl start_app

  rabbitmq-init 
}


rabbitmq-init(){
  sudo rabbitmqctl delete_user guest
  local ids="0 1 2 3"
  for id in $ids ; do
    rabbitmq-register-  $id
  done 
}




rabbitmq-install-yum(){

   #redhat-
   #redhat-epel    ## hookup epel repo 

   sudo yum install erlang  

   ## for RHEL4   
   ##sudo rpm -Uvh http://www.rabbitmq.com/releases/rabbitmq-server/v1.7.0/rabbitmq-server-1.7.0-1.i386.rpm
   ## after enabling EPEL (see :  redhat- ; redhat-epel4 ) ... can just do 

   sudo yum install rabbitmq-server


}

rabbitmq-install-port(){
  sudo port install rabbitmq-server
}


rabbitmq-open-ip(){
  local ip=$1
  private-
  iptables-
  IPTABLES_PORT=$(local-port rabbitmq) iptables-webopen-ip $ip 
}

rabbitmq-hrl(){
  rpm -ql rabbitmq-server | grep rabbit.hrl
}



rabbitmq-open(){
   local tag=$1
   $FUNCNAME-ip $(local-tag2ip $tag)
}

rabbitmq-wideopen(){
  iptables-
  IPTABLES_PORT=$(local-port rabbitmq) iptables-webopen
}



## for info only ... the server was installed from EPEL repo

rabbitmq-server-dir(){ echo $(rabbitmq-dir)/rabbitmq-server ; }
rabbitmq-hg(){  echo http://hg.rabbitmq.com ; }
rabbitmq-server-cd(){  cd $(rabbitmq-server-dir) ; }
rabbitmq-server-get(){
  local dir=$(rabbitmq-dir)
  mkdir -p $dir && cd $dir
  hg clone $(rabbitmq-hg)/rabbitmq-server
}

rabbitmq-info(){
   yum --enablerepo=epel info rabbitmq-server
}

rabbitmq-smry()
{
   local xs="exchanges queues connections bindings users"
   local x
   for x in $xs ; do
        echo $x
        rabbitmq-$x
   done
}



rabbitmq-fields(){    ## from the rabbitmqctl usage message 
   case $1 in
      exchanges) echo name type durable auto_delete arguments  ;;
         queues_cms01) echo name durable auto_delete arguments node messages_ready  messages_unacknowledged messages_uncommitted messages acks_uncommitted consumers transactions memory ;; 
         queues) echo name durable auto_delete arguments messages_ready  messages_unacknowledged messages_uncommitted messages acks_uncommitted consumers transactions memory ;; 
    connections_cms01) echo node address port peer_address peer_port state channels user vhost timeout frame_max recv_oct recv_cnt send_oct send_cnt send_pend ;; 
    connections) echo address port peer_address peer_port state channels user vhost timeout frame_max recv_oct recv_cnt send_oct send_cnt send_pend ;; # client_properties 
       bindings) echo exchange_name routing_key queue_name arguments ;;
   esac
}
rabbitmq-exchanges(){   rabbitmq-list ${FUNCNAME/*-/} $* ; }  
rabbitmq-queues(){      rabbitmq-list ${FUNCNAME/*-/} $* ; }  
rabbitmq-connections(){ rabbitmq-list ${FUNCNAME/*-/} $* ; }  
rabbitmq-bindings(){    rabbitmq-list ${FUNCNAME/*-/} $* ; }  
rabbitmq-users(){       rabbitmq-list ${FUNCNAME/*-/} $* ; }  

rabbitmq-list(){
    local ty=${1:-queues}
    shift
    local args
    [ "$#" == "0" ] && args=$(rabbitmq-fields $ty) || args=$*
    echo $args  | rabbitmq-tabulate
    sudo rabbitmqctl -q list_$ty $args | rabbitmq-tabulate 
}

rabbitmq-tabulate(){
  perl -n -e '@a = split("\t") ;  @a = split(" ") if($#a == 0);  printf "%-35s "." %-10s " x ($#a - 1) ."\n", @a  ; ' -
}




## this is using py-amqplib see carrot- for a slightly higher level interface 

rabbitmq-ex-preq(){       
    pip install amqplib 
    private-
    private-py-install
}
rabbitmq-ex-dir(){         echo $(env-home)/messaging/rabbits_and_warrens ; } 
rabbitmq-ex-cd(){          cd $(rabbitmq-ex-dir) ; }
rabbitmq-ex-consumer(){    python $(rabbitmq-ex-dir)/amqp_consumer.py $* ; }
rabbitmq-ex-publisher(){   python $(rabbitmq-ex-dir)/amqp_publisher.py $* ; }






rabbitmq-umbrella-dir(){ echo $(rabbitmq-dir)/rabbitmq-public-umbrella ;  }
rabbitmq-umbrella-cd(){  cd $(rabbitmq-umbrella-dir) ; }
rabbitmq-umbrella-get(){
   rabbitmq-cd
   local dir=$(rabbitmq-dir)
   mkdir -p $dir && cd $dir
   hg clone http://hg.rabbitmq.com/rabbitmq-public-umbrella
}

rabbitmq-umbrella-make(){
   rabbitmq-umbrella-cd
   ## check out subprojects into the umbrella
   make co

}
