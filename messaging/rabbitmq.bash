# === func-gen- : messaging/rabbitmq fgp messaging/rabbitmq.bash fgn rabbitmq fgh messaging
rabbitmq-src(){      echo messaging/rabbitmq.bash ; }
rabbitmq-source(){   echo ${BASH_SOURCE:-$(env-home)/$(rabbitmq-src)} ; }
rabbitmq-vi(){       vi $(rabbitmq-source) ; }
rabbitmq-env(){      elocal- ; }
rabbitmq-usage(){
  cat << EOU
     rabbitmq-src : $(rabbitmq-src)
     rabbitmq-dir : $(rabbitmq-dir)


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




rabbitmq-c-dir(){ echo $(rabbitmq-dir)/rabbitmq-c ; }
rabbitmq-c-cd(){  cd $(rabbitmq-c-dir) ; }

rabbitmq-c-build(){
   rabbitmq-c-preq
   rabbitmq-c-get
   rabbitmq-c-make
}


rabbitmq-c-preq(){
   pip install simplejson 
}

rabbitmq-c-get(){
  local dir=$(rabbitmq-dir)
  mkdir -p $dir && cd $dir
  hg clone http://hg.rabbitmq.com/rabbitmq-c

  ## rabbitmq-c expects a codegen dir containing : amqp-0.8.json and amqp_codegen.py
  #cd rabbitmq-c
  #hg clone http://hg.rabbitmq.com/rabbitmq-codegen codegen
}

rabbitmq-c-make(){
  rabbitmq-c-cd
  autoreconf -i
  ./configure 
  make 
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
    echo $args 
    sudo rabbitmqctl -q list_$ty $args
}




rabbitmq-ex-preq(){       

    pip install amqplib 
    
    private-
    private-py-install

}
rabbitmq-ex-dir(){         echo $(env-home)/messaging/rabbits_and_warrens ; } 
rabbitmq-ex-cd(){          cd $(rabbitmq-ex-dir) ; }
rabbitmq-ex-consumer(){    python $(rabbitmq-ex-dir)/amqp_consumer.py $* ; }
rabbitmq-ex-publisher(){   python $(rabbitmq-ex-dir)/amqp_publisher.py $* ; }



