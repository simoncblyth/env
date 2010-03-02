# === func-gen- : messaging/rabbitmq fgp messaging/rabbitmq.bash fgn rabbitmq fgh messaging
rabbitmq-src(){      echo messaging/rabbitmq.bash ; }
rabbitmq-source(){   echo ${BASH_SOURCE:-$(env-home)/$(rabbitmq-src)} ; }
rabbitmq-vi(){       vi $(rabbitmq-source) ; }
rabbitmq-env(){      elocal- ; }
rabbitmq-usage(){
  cat << EOU
     rabbitmq-src : $(rabbitmq-src)
     rabbitmq-dir : $(rabbitmq-dir)

     rabbitmq-status
     rabbitmq-start
     rabbitmq-stop
           ini controls of the rabbitmq node


     rabbitmq-exchanges
     rabbitmq-queues
     rabbitmq-bindings
     rabbitmq-connections
               interrogate the server to provide listings of diagnostic fields, 
               provide fieldname arguments to restrict to a subset of the fields 

  On OSX :
      http://trac.macports.org/browser/trunk/dports/net/rabbitmq-server/Portfile
      http://trac.macports.org/browser/trunk/dports/lang/erlang/Portfile
           Why does erlang depend on wxWidgets ?
           http://www.erlang.org

   man rabbitmq-server
   man rabbitmq.conf
   man rabbitmq-multi
   man rabbitmqctl



     rabbitmq-c-build
         build the rabbitmq C client 
   
     rabbitmq-c-sendstring 


       

     rabbitmq-ex-consumer
     rabbitmq-ex-publisher
            try py-ampqlib based consumer and publisher




     http://www.rabbitmq.com/plugin-development.html#getting-started 

     rabbitmq-umbrella-get
            umbrella is simply just a Makefile, that can checkout all rabbitmq-* and build

     rabbitmq-umbrella-make



EOU
}
rabbitmq-dir(){ echo $(local-base)/env/messaging ; }
rabbitmq-cd(){  cd $(rabbitmq-dir); }
rabbitmq-mate(){ mate $(rabbitmq-dir) ; }

rabbitmq-logdir(){   echo /var/log/rabbitmq ; }
rabbitmq-logpath(){  echo $(rabbitmq-logdir)/rabbit.log ; }
rabbitmq-tail(){   sudo tail -f $(rabbitmq-logpath) ; }
rabbitmq-confpath(){ echo /etc/rabbitmq/rabbitmq.conf ; }
rabbitmq-edit(){     sudo vi $(rabbitmq-confpath) ; }

rabbitmq-inipath(){ echo /etc/init.d/rabbitmq-server ; }
rabbitmq-ini(){     sudo $(rabbitmq-inipath) $* ; }

rabbitmq-start(){   rabbitmq-ini start ; }
rabbitmq-status(){  rabbitmq-ini status ; }
rabbitmq-stop(){    rabbitmq-ini stop ; }


rabbitmq-install-yum(){

   #redhat-
   #redhat-epel    ## hookup epel repo 

   sudo yum install erlang  

   ## for RHEL4   
   sudo rpm -Uvh http://www.rabbitmq.com/releases/rabbitmq-server/v1.7.0/rabbitmq-server-1.7.0-1.i386.rpm
}

rabbitmq-install-port(){
  sudo port install rabbitmq-server
}


rabbitmq-open-ip(){
  local ip=$1
  private-
  iptables-
  IPTABLES_PORT=$(private-val AMQP_PORT) iptables-webopen-ip $ip 
}
rabbitmq-open(){
   local tag=$1
   $FUNCNAME-ip $(local-tag2ip $tag)
}


#rabbitmq-conf(){
#  pkgr-
#  case $(pkgr-cmd) in 
#     yum) echo 
#    port) 
#  esac
#}


rabbitmq-hg(){  echo http://hg.rabbitmq.com ; }


rabbitmq-server-dir(){ echo $(rabbitmq-dir)/rabbitmq-server ; }
rabbitmq-server-cd(){  cd $(rabbitmq-server-dir) ; }
rabbitmq-server-get(){
  local dir=$(rabbitmq-dir)
  mkdir -p $dir && cd $dir
  hg clone $(rabbitmq-hg)/rabbitmq-server
}


rabbitmq-c-dir(){ echo $(rabbitmq-dir)/rabbitmq-c ; }
rabbitmq-c-libdir(){ echo $(rabbitmq-c-dir)/librabbitmq/.libs ; }
rabbitmq-c-cd(){  cd $(rabbitmq-c-dir) ; }

rabbitmq-c-build(){
   rabbitmq-c-preq

   rabbitmq-codegen-get
   rabbitmq-c-get
   rabbitmq-c-make
}


rabbitmq-c-preq(){
   pip install simplejson 
}

rabbitmq-c-get(){
  local dir=$(rabbitmq-dir)
  mkdir -p $dir && cd $dir
  hg clone $(rabbitmq-hg)/rabbitmq-c
}

rabbitmq-c-make(){
  rabbitmq-c-cd
  rabbitmq-c-kludge

  autoreconf -i
  autoconf
  ./configure 

  ## avoid hardcoded attempt to use python2.5
  make PYTHON=python
}

rabbitmq-c-kludge(){
  perl -pi -e "s,(sibling_codegen_dir=).*,\$1\"$(rabbitmq-codegen-dir)\"," configure.ac
}





rabbitmq-c-exepath(){ echo $(rabbitmq-c-dir)/examples/amqp_$1 ; }

rabbitmq-c-exchange(){ echo ${RMQC_EXCHANGE:-amq.direct} ; }
rabbitmq-c-queue(){    echo ${RMQC_QUEUE:-test queue} ; }
rabbitmq-c-key(){      echo ${RMQC_KEY:-test queue} ; }

rabbitmq-c-consumer(){
   local msg="=== $FUNCNAME :"
   local exe=$(rabbitmq-c-exepath consumer)
   [ ! -x "$exe" ] && echo $msg ABORT no executable at $exe && return 1 
   private-
   local host=$(private-val AMQP_SERVER) 
   local port=$(private-val AMQP_PORT) 
   local cmd="$exe $host $port " 
   echo $msg $cmd CAUTION hardcoded : exchange  \"amq.direct\" and key  \"test queue\"
   eval $cmd
}

rabbitmq-c-sendstring(){
   local msg="=== $FUNCNAME :"
   local exe=$(rabbitmq-c-exepath sendstring)
   [ ! -x "$exe" ] && echo $msg ABORT no executable at $exe && return 1 
   
   private-
   local host=$(private-val AMQP_SERVER) 
   local port=$(private-val AMQP_PORT) 
   local exchange=$(rabbitmq-c-exchange)
   local routingkey=$(rabbitmq-c-queue)
   local messagebody="$(hostname) $(date)"
   local cmd="$exe $host $port $exchange \"$routingkey\" \"$messagebody\""
   echo $msg $cmd
   eval $cmd
}

rabbitmq-c-listen(){
   local msg="=== $FUNCNAME :"
   local exe=$(rabbitmq-c-exepath listen)
   [ ! -x "$exe" ] && echo $msg ABORT no executable at $exe && return 1 
   
   private-
   local host=$(private-val AMQP_SERVER) 
   local port=$(private-val AMQP_PORT) 
   local exchange=$(rabbitmq-c-exchange)
   local routingkey=$(rabbitmq-c-queue)
   local cmd="$exe $host $port $exchange \"$routingkey\" "
   echo $msg $cmd
   eval $cmd

}

rabbitmq-c-usage(){
   local files="amqp_sendstring.c example_utils.c example_utils.h"
   cd $(env-home)/notifymq 
   local file ; for file in $files ; do
     [ ! -f "$file" ] &&  cp $(rabbitmq-c-dir)/examples/$file . || echo $msg $file already present in $PWD
   done
}



rabbitmq-codegen-dir(){ echo $(rabbitmq-dir)/rabbitmq-codegen ; }
rabbitmq-codegen-cd(){  cd $(rabbitmq-codegen-dir) ; }
rabbitmq-codegen-get(){
  local dir=$(rabbitmq-dir)
  mkdir -p $dir && cd $dir
  hg clone http://hg.rabbitmq.com/rabbitmq-codegen
}








rabbitmq-fields(){    ## from the rabbitmqctl usage message 
   case $1 in
      exchanges) echo name type durable auto_delete arguments  ;;
         queues) echo name durable auto_delete arguments node messages_ready  messages_unacknowledged messages_uncommitted messages acks_uncommitted consumers transactions memory ;; 
    connections) echo node address port peer_address peer_port state channels user vhost timeout frame_max recv_oct recv_cnt send_oct send_cnt send_pend ;; 
       bindings) echo exchange_name routing_key queue_name arguments ;;
   esac
}
rabbitmq-exchanges(){   rabbitmq-list ${FUNCNAME/*-/} $* ; }  
rabbitmq-queues(){      rabbitmq-list ${FUNCNAME/*-/} $* ; }  
rabbitmq-connections(){ rabbitmq-list ${FUNCNAME/*-/} $* ; }  
rabbitmq-bindings(){    rabbitmq-list ${FUNCNAME/*-/} $* ; }  

rabbitmq-list(){
    local ty=${1:-queues}
    shift
    local args
    [ "$#" == "0" ] && args=$(rabbitmq-fields $ty) || args=$*
    echo $args  | rabbitmq-tabulate
    sudo rabbitmqctl -q list_$ty $args | rabbitmq-tabulate 
}

rabbitmq-tabulate(){
  perl -n -e '@a = split("\t") ;  @a = split(" ") if($#a == 0);  printf "%-20s "." %-10s " x ($#a - 1) ."\n", @a  ; ' -
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
