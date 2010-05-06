# === func-gen- : messaging/carrot/carrot fgp messaging/carrot/carrot.bash fgn carrot fgh messaging/carrot
carrot-src(){      echo messaging/carrot/carrot.bash ; }
carrot-source(){   echo ${BASH_SOURCE:-$(env-home)/$(carrot-src)} ; }
carrot-vi(){       vi $(carrot-source) ; }
carrot-env(){      elocal- ; }
carrot-usage(){
  cat << EOU
     carrot-src : $(carrot-src)
     carrot-dir : $(carrot-dir)

     carrot-consumer
           create a consumer and wait for messages, usage :
               carrot-consumer --help
               carrot-consumer -q theq -e thexchange -k dekey
           OR just use defaults :
               carrot-consumer  

     carrot-publisher
            publish message via the queue 

     carrot-connection
            test the connection to the MQ configured via private AMQP_... 
            the corresponding python module is used by both consumer and publisher

EOU
}

carrot-preq(){
   pip install carrot
   pip install python-cjson
}

carrot-dir(){ echo $(env-home)/messaging/carrot ; }
carrot-cd(){  cd $(carrot-dir); }
carrot-mate(){ mate $(carrot-dir) ; }

carrot-consumer(){    python $(carrot-dir)/carrot_consumer.py $* ;  }
carrot-publisher(){   python $(carrot-dir)/carrot_publisher.py $* ;  }
carrot-connection(){  python $(carrot-dir)/carrot_connection.py $* ;  }


