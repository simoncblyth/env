# === func-gen- : messaging/camqadm fgp messaging/camqadm.bash fgn camqadm fgh messaging
camqadm-src(){      echo messaging/camqadm.bash ; }
camqadm-source(){   echo ${BASH_SOURCE:-$(env-home)/$(camqadm-src)} ; }
camqadm-vi(){       vi $(camqadm-source) ; }
camqadm-env(){      elocal- ; }


camqadm-svi(){ vi $(python-site)/celery/bin/camqadm.py ; }
camqadm-usage(){
  cat << EOU
     camqadm-src : $(camqadm-src)
     camqadm-dir : $(camqadm-dir)

     camqadm is an AMQP command line client that comes with recent celery (eg 2.1.3)
     to get it upgrade celery with 

           pip install --upgrade celery   
	   pip install --upgrade celery==dev   
   
    or check your version with 
           pip search celery 

     Unfortunately both these exhibit the below issue so get 
     latest from github :

          celery-
          celery-get
          celery-build
          
    ISSUES 
      Trying to use yes/no options gives "too many values to unpack errors"
          https://github.com/ask/celery/issues/issue/257


    http://ask.github.com/celery/reference/celery.bin.camqadm.html
     2.1 docs 
          http://celeryq.org/docs/userguide/routing.html#hands-on-with-the-api

     dev docs
         http://ask.github.com/celery/userguide/routing.html#hands-on-with-the-api
    
     API refernce 
         http://ask.github.com/celery/reference/celery.bin.camqadm.html

        camqadm--
             invoke camqadm CLI with messaging/camqadm on PYTHONPATH
             in order to find the celeryconfig.py file 
             which uses the private config 


    == pika talking to camqadm ? ==


    Trying to talk from pika to camqadm ... run into different
    queue parameter defaults that cause error on pika-consume

        pika-consume -v -q q
             "NOT_ALLOWED - parameters for queue 'q' in vhost '/' not equivalent" 

    Default queue parameters in camqadm :
        queue.declare queue passive:no durable:no exclusive:no auto_delete:no 
 

    Create+bind queue and exchange with camqadm :
        
        exchange.declare x fanout
        queue.declare q         
        queue.bind q x dummy.string

    and then send from pika :
        pika-send -v -x x -k dummy.string 

    can be picked up with camqadm ...

    43> basic.get q
{'body': 'test message from /data/env/local/env/home/messaging/pika/send.py on cms01.phys.ntu.edu.tw ',
 'delivery_info': {'delivery_tag': 9,
                   'exchange': u'x',
                   'message_count': 0,
                   'redelivered': False,
                   'routing_key': u'dummy.string'},
 'properties': {'content_type': u'text/plain', 'delivery_mode': 2}}


   == use camqadm to probe q parameters/occupancy  ==


2> queue.declare q no no no no
(530, u"NOT_ALLOWED - parameters for queue 'q' in vhost '/' not equivalent", (50, 10), 'Channel.queue_declare')

3> queue.declare q yes no no no
-> connecting to amqplib://guest@aberdeen.phy.cuhk.edu.hk:5672/.
-> connected.
ok. queue:q messages:9 consumers:0.

   
       * determines that "passive" is true ...

  Now can talk with pika via the CUHK server thanks to :
      *  "-p" option for passive 
      * checkout of pika 0_9_1 branch with pika-091

       pika-consume -v -p -q q
       pika-send    -v -x x

  And with camqadm too :

      camqadm-- basic.publish hello-from-camqadm x k



EOU
}
camqadm-dir(){ echo $(local-base)/env/messaging/messaging-camqadm ; }
camqadm-cd(){  cd $(camqadm-dir); }
camqadm-mate(){ mate $(camqadm-dir) ; }
camqadm-get(){
   local dir=$(dirname $(camqadm-dir)) &&  mkdir -p $dir && cd $dir
}


camqadm--(){
   PYTHONPATH=$(env-home)/messaging/camqadm camqadm $* 
}
