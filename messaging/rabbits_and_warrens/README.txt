To test the RabbitMQ server ... 

  1) start the consumer 
{{{
   cd ~/e/messaging/rabbits_and_warrens 
   python amqp_consumer.py 
}}}

  2) publish messages from the same or remote nodes 
{{{
   cd ~/e/messaging/rabbits_and_warrens
   python amqp_publisher.py hello from $(hostname) at $(date) 
}}}
   which should show up on the consumer


Further reading ...
   rabbitmq-vi
   carrot-vi



