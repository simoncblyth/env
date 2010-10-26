# === func-gen- : nodejs/rabbitjs fgp nodejs/rabbitjs.bash fgn rabbitjs fgh nodejs
rabbitjs-src(){      echo nodejs/rabbitjs.bash ; }
rabbitjs-source(){   echo ${BASH_SOURCE:-$(env-home)/$(rabbitjs-src)} ; }
rabbitjs-vi(){       vi $(rabbitjs-source) ; }
rabbitjs-env(){      
    elocal-  
    nodejs-
    nodejs-path
}
rabbitjs-usage(){
  cat << EOU
     rabbitjs-src : $(rabbitjs-src)
     rabbitjs-dir : $(rabbitjs-dir)


     http://github.com/squaremo/rabbit.js


    Using squaremo node-amqp fork , see nodeamqp-




    socketio.js (coordination hub)
        createServer that serves the *.html 
        extend server with 
          socks.listen(socketserver, {
            'requests': ['rep', 'req'],
            'chat': ['pub', 'sub']
           });
        which is implemented in sockets.js


    sockets.js
         establish amqp connection and dispatches messages to 
         various socket types
 

    *.html
        looks like demonstration of socket.io usage 
        ... nothing rabbit specific 

    messages.js
        * MessageStream.send encodes message length in 4-word header
        * reads on stream until length(from 4-word header) sufficient to make a message
  

== Dependency tree between the .js   (grep require *.js) ==

    ./socketio : http url fs ./Socket.IO-node/lib/socket.io sys ./sockets
    ./sockets : ./node-amqp sys  
    ./socketserver : net ./messages ./sockets 
    ./testsrv : ./messages net 
    ./messages : buffer events sys

== fix require paths for flat list of repo checkouts  ==

  * socketio.js
-var io = require('./Socket.IO-node/lib/socket.io');
+var io = require('../socket.io-node/lib/socket.io');

  * sockets.js
-var amqp = require('./node-amqp/');
+var amqp = require('../node-amqp/');

== try {{{node socketio.js}}} /pubsub.html demo ==

    * get "Stream not writable" errors, until fix the config of amqp backend in sockets.js

-var connection = amqp.createConnection({'host': '127.0.0.1', 'port': 5672});
+var cfg = { host:process.env["AMQP_LOCAL_SERVER"], port:process.env["AMQP_LOCAL_PORT"], login:process.env["AMQP_LOCAL_USER"] , password:process.env["AMQP_LOCAL_PASSWORD"], vhost:process.env["AMQP_LOCAL_VHOST"]   };
+var connection = amqp.createConnection(cfg);

== test sending message to the browsers from bunny AMQP client  ==

[blyth@belle7 rabbit.js]$ bunny-- LOCAL_
belle7.nuu.edu.tw./: send_message chat:jello from bunny

== test with mobile safari (iOS 3.1.3 using transport xhr-multipath ) ==

  * succeeds to send receive/several messages  
  * but seems prone to disconnecting after a few minutes ...

25 Oct 20:52:01 - Client 2532701303716749 disconnected
25 Oct 20:52:04 - Couldnt find client with session id "2532701303716749"




EOU
}
rabbitjs-dir(){ echo $(local-base)/env/nodejs/rabbit.js ; }
rabbitjs-cd(){  cd $(rabbitjs-dir); }
rabbitjs-mate(){ mate $(rabbitjs-dir) ; }
rabbitjs-get(){
   local dir=$(dirname $(rabbitjs-dir)) &&  mkdir -p $dir && cd $dir

   git clone http://github.com/squaremo/rabbit.js.git
}

rabbitjs-run(){
   rabbitjs-cd

   nodeamqp-
   nodeamqp-cfg-export
   node socketio.js
  
}


