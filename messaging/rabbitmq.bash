# === func-gen- : messaging/rabbitmq fgp messaging/rabbitmq.bash fgn rabbitmq fgh messaging
rabbitmq-src(){      echo messaging/rabbitmq.bash ; }
rabbitmq-source(){   echo ${BASH_SOURCE:-$(env-home)/$(rabbitmq-src)} ; }
rabbitmq-vi(){       vi $(rabbitmq-source) ; }
rabbitmq-env(){      elocal- ; }
rabbitmq-usage(){
  cat << EOU
     rabbitmq-src : $(rabbitmq-src)
     rabbitmq-dir : $(rabbitmq-dir)


EOU
}
rabbitmq-dir(){ echo $(local-base)/env/messaging/messaging-rabbitmq ; }
rabbitmq-cd(){  cd $(rabbitmq-dir); }
rabbitmq-mate(){ mate $(rabbitmq-dir) ; }


rabbitmq-install(){

   #redhat-
   #redhat-epel    ## hookup epel repo 

   sudo yum install erlang  

   ## for RHEL4   
   sudo rpm -Uvh http://www.rabbitmq.com/releases/rabbitmq-server/v1.7.0/rabbitmq-server-1.7.0-1.i386.rpm


}


