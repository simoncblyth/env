

  libnotifymq provides an interface to the usage of the rabbitmq-c 
  AMQP producer/consumer from root (pyROOT/cint/compiled)
  allowing TObjects to be sent and received via a RabbitMQ server 

  The use of a separate message queue server decouples the production and 
  consumption of TObjects, allowing asynchronous communication
  (analogous to email) between processes

     src/
     include/   
              Sources and headers for the library 


     tests/
              pyROOT and cint tests using the lib 
       

     ./
              Short testing mains are here 

                 mq_sendstring
                       note that the string is wrapped into MyTMessage
                       (use SendRaw for sending to non-root consumers, as
                        when using JSON to communicate with web apps  )

                 mq_monitor
                       monitor is started in separate thread preventing blocking, the monitor
                       thread is kept as simple as possible ... avoid use of ROOT objects 

                 mq_threaded
                        initial investigations of using threading , this
                        approach suffers from deadlocks as too much is done in the monitor thread 
 
                 mq_consumebutes
                       deprecated approach using MQ::Wait that blocks on notifymq_basic_consume
                       while waiting for messages

                 mq_mapfile
                       deprecated approach that receives TObjects and writes into TMapFile 
                       for consumption by a separate process 

 




