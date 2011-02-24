
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





== pikamq dev notes ==


Attempt to introduce threading to the pika examples
BUT failed to doit with threads alone, so using 
multiprocessing to workaround pika problems with 
consumer and publisher in the same process
    
    * http://docs.python.org/library/multiprocessing.html#module-multiprocessing 
    * http://www.ibm.com/developerworks/aix/library/au-multiprocessing/
    * https://github.com/tonyg/pika#readme
    * http://tonyg.github.com/pika/
    * http://www.amqp.org/confluence/download/attachments/720900/amqp0-9-1.pdf
    * http://www.rabbitmq.com/tutorial-three-python.html

Notes 
   * when changing parameters of an exchnage/queue do a rabbitmq-reset to remove 
     preexisting entities otherwise the callbacks never gets back to you 

   Declaring exchange: server channel error 406, message: PRECONDITION_FAILED - cannot redeclare exchange 'abt' in vhost '/' 
     with different type, durable, internal or autodelete value


Test with :
   pika- ; pika-i
   ipython> run mon.py

Publish from another process with :

   (mq)simon:pika blyth$ camqadm
    /Users/blyth/v/mq/lib/python2.6/site-packages/celery-2.2.4-py2.6.egg/celery/loaders/default.py:54: NotConfigured: No 'celeryconfig' module found! Please make sure it exists and is available to Python.
  "is available to Python." % (configname, )))
   -> connecting to amqplib://guest@localhost:5672/.
   -> connected.
   1> basic.publish aaa xmonf test
   ok.
   2> basic.publish bbb xmonf test
   ok.
   3> basic.publish ccc xmonf test
   ok.
   4> 

to start from a clean slate :

   sudo rabbitmqctl stop_app
   sudo rabbitmqctl reset   
   sudo rabbitmqctl start_app

ISSUES...

  1) duplication / round-robinning  
     getting messages twice and missing every other ???
 
     caused by double connected consumer
     which arose due to a one_shot=False for the default pika on_open callback ... 

     after fix this ... still failing to get a publisher and consumer to 
     co-exist within the same process  

NEXT

     * breaking up the monolith for reuse/flexibility 
          * splitting ROOT and pika dependencies

         divide into separate pots...

              pika/rabbitmq wiring 
              threading/multiprocessing
              msg handling
              root
 

     * aping the rootmq gMQ API
     * distributed testing
     * GUI integration / timers
     * 



