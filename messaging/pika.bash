# === func-gen- : messaging/pika fgp messaging/pika.bash fgn pika fgh messaging
pika-src(){      echo messaging/pika.bash ; }
pika-source(){   echo ${BASH_SOURCE:-$(env-home)/$(pika-src)} ; }
pika-vi(){       vi $(pika-source) ; }
pika-env(){      elocal- ; }
pika-usage(){
  cat << EOU
     pika-src : $(pika-src)
     pika-dir : $(pika-dir)
     
     http://github.com/tonyg/pika

     To test operation ... 
         1) in one session : pika-consume
         2) in another     : pika-send

      The sessions can be on different machines but the 
      AMQP_ config for the sessions has to point 
      to the same vhost in the same server (or erlang cluster i suppose)

     Usage example ...

          pika-consume --help
          pika-consume -x testolive
             fails noisily if exchange does not exist 


EOU
}
pika-dir(){ echo $(local-base)/env/messaging/pika ; }
pika-cd(){  cd $(pika-dir); }
pika-mate(){ mate $(pika-dir) ; }
pika-get(){
   local dir=$(dirname $(pika-dir)) &&  mkdir -p $dir && cd $dir
   git clone http://github.com/tonyg/pika.git
}

pika-ln(){
  python-
  python-ln $(pika-dir)/pika
}

pika-send(){    python $(env-home)/messaging/pika/send.py $* ; }
pika-consume(){ python $(env-home)/messaging/pika/consume.py $* ; }

