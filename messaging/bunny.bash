# === func-gen- : messaging/bunny fgp messaging/bunny.bash fgn bunny fgh messaging
bunny-src(){      echo messaging/bunny.bash ; }
bunny-source(){   echo ${BASH_SOURCE:-$(env-home)/$(bunny-src)} ; }
bunny-srcdir(){   echo $(dirname $(bunny-source)) ; }
bunny-vi(){       vi $(bunny-source) ; }
bunny-env(){      elocal- ; }
bunny-usage(){
  cat << EOU
     bunny-src : $(bunny-src)
     bunny-dir : $(bunny-dir)

     http://github.com/bkjones/bunny
         cmd.Cmd based interactive python client to rabbitmq, 
         based on python module amqplib 

    Extensive usage examples at :
        http://dayabay.phys.ntu.edu.tw/tracs/env/wiki/RabbitMQFanout


     bunny-build
        -get and -kludge

     bunny-kludge
        fix py2.6isms   
            "except ValueError as out:"
         -->"except ValueError , out:"

     bunny--original
        run interactive client

     bunny-- <name>
        run monkey patched mybunny.py that has the private_connect command 
        which uses the private- vars and which auto-connects on startup

        Usage example : 

           bunny-- OTHER_
           Trying connect to cms01.phys.ntu.edu.tw:/ as abtviz ... name:OTHER_ 
           Success!
           cms01.phys.ntu.edu.tw./: 
           cms01.phys.ntu.edu.tw./: create_exchange name=sorting_room
           No type - using 'direct'
           cms01.phys.ntu.edu.tw./: create_queue po_box
           cms01.phys.ntu.edu.tw./: create_binding exchange=sorting_room queue=po_box
           cms01.phys.ntu.edu.tw./: send_message sorting_room:hello from my bunny


        Using parameter set AMQP_OTHER_SERVER/USER/PASSWORD/VHOST   

        As the exange and queue match those of the amqp_consumer.py example ...
            simon:rabbits_and_warrens blyth$ python amqp_consumer.py 
        that consumer will get the message 
        
        
        Tried deleting a queue as its being written to by sendobj .... it got deleted and no errors from sendobj 
        
        simon:e blyth$ bunny--       
        Trying connect to cms01.phys.ntu.edu.tw:/ as user abtviz ... using private-vals AMQP_<name>SERVER/USER/PASSWORD/VHOST where name: 
        Success!
        cms01.phys.ntu.edu.tw./: delete_queue N
        cms01.phys.ntu.edu.tw./:
        
        
        Following a config change to make the queue durable ... 
           * restarting sendobj works but the queue did not become durable 
           * have to delete the queue first      
        
        


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

bunny--original(){
   python $(bunny-dir)/bunny.py
}
bunny--(){
   PYTHONPATH=$(bunny-dir) python $(bunny-srcdir)/mybunny.py $*
}
bunny-kludge(){
   perl -pi -e s'@as out:@, out:@g' $(bunny-dir)/bunny.py
   perl -pi -e 's,^shell,#shell,g' $(bunny-dir)/bunny.py    ## comment last 2 lines that prevent Monkey patching 
}


bunny-amqplib(){
   python-
   mate $(python-site)/amqplib/client_0_8/
}




