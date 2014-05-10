CZMQ HelloWorld Client/Server
=============================

Directly from client to server
--------------------------------

NB this failed to work but gave no error on port 5000. It works OK on 5001 and 5002.

::

    delta:~ blyth$ czmq_server tcp://*:5001 
    INFO: czmq_server server echoing to:[tcp://*:5001]   
    Server got request: yo
    Server got request: yo
    Server got request: yo
    Server got request: yo

::

    delta:~ blyth$ czmq_client tcp://localhost:5001 yo
    INFO: czmq_client client repeatedly sending to:[tcp://localhost:5001] message string [yo] 
    Client got reply: yo
    Client got reply: yo
    Client got reply: yo


Client to Server via Queue Device
----------------------------------------

::

    delta:~ blyth$ czmq_client tcp://localhost:5001 yo
    INFO: czmq_client client repeatedly sending to:[tcp://localhost:5001] message string [yo] 

::

    delta:~ blyth$ czmq_server tcp://*:5002
    INFO: czmq_server server echoing to:[tcp://*:5002]   




