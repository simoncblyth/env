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

pika-ex-receive(){
  python $(env-home)/messaging/pika/my_demo_receive.py
}

pika-ex-send(){
  python $(env-home)/messaging/pika/my_demo_send.py
}

