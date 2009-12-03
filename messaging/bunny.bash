# === func-gen- : messaging/bunny fgp messaging/bunny.bash fgn bunny fgh messaging
bunny-src(){      echo messaging/bunny.bash ; }
bunny-source(){   echo ${BASH_SOURCE:-$(env-home)/$(bunny-src)} ; }
bunny-vi(){       vi $(bunny-source) ; }
bunny-env(){      elocal- ; }
bunny-usage(){
  cat << EOU
     bunny-src : $(bunny-src)
     bunny-dir : $(bunny-dir)

     http://github.com/bkjones/bunny
         interractive python client to rabbitmq, based on python module amqplib 


     bunny-build
        -get and -kludge

     bunny-kludge
        fix py2.6isms   
            "except ValueError as out:"
         -->"except ValueError , out:"

     bunny--
        run interactive client


EOU
}
bunny-dir(){ echo $(local-base)/env/messaging/bunny ; }
bunny-cd(){  cd $(bunny-dir); }
bunny-mate(){ mate $(bunny-dir) ; }
bunny-get(){
   local dir=$(dirname $(bunny-dir)) &&  mkdir -p $dir && cd $dir
   git clone git://github.com/bkjones/bunny.git
}

bunny-build(){
   bunny-get
   bunny-kludge
}


bunny--(){
   python $(bunny-dir)/bunny.py
}

bunny-kludge(){
   perl -pi -e s'@as out:@, out:@g' $(bunny-dir)/bunny.py
}
