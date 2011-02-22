
  consume.py
  send.py
               hail from early pika with Asyncore



  timed_receive.py 
       based on  /usr/local/env/messaging/pika/examples/timed_receive.py              




== python threading ==

  * http://www.ibm.com/developerworks/aix/library/au-threadingpython/


The virtual python should be equipped with :
      * ipython  ... pip install --upgrade ipython
              /Users/blyth/v/mq/bin/ipython      
                     CAUTION USE FULL PATH TO IPYTHON 

      * camaqdm  ... from celery 

             camqadm-
             camqadm--



   http://lists.rabbitmq.com/pipermail/rabbitmq-discuss/2011-February/011184.html
      maybe the normal thing to do is share a connection ... and have a channel for each thread

      nope .. pika FAQ says :
   
        Pika does not have any notion of threading in the code. 
        If you want to use Pika with threading, make sure you have a Pika connection per thread, 
        created in that thread. It is not safe to share one Pika connection across threads.



  camqadm kills the consumer with ...


mq)simon:e blyth$ camqadm
/Users/blyth/v/mq/lib/python2.6/site-packages/celery-2.2.4-py2.6.egg/celery/loaders/default.py:54: NotConfigured: No 'celeryconfig' module found! Please make sure it exists and is available to Python.
  "is available to Python." % (configname, )))
-> connecting to amqplib://guest@localhost:5672/.
-> connected.
2> basic.publish a "" test
ok.


3> basic.publish a "" test
ok.
4> basic.publish b "" test
ok.


    .... but the toubles seem not to happen if do not try to produce and consume
    from the same process (in different threads)
   
     ...... smth is being shared that should not be ????


