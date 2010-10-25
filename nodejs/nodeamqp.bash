# === func-gen- : nodejs/nodeamqp fgp nodejs/nodeamqp.bash fgn nodeamqp fgh nodejs
nodeamqp-src(){      echo nodejs/nodeamqp.bash ; }
nodeamqp-source(){   echo ${BASH_SOURCE:-$(env-home)/$(nodeamqp-src)} ; }
nodeamqp-vi(){       vi $(nodeamqp-source) ; }
nodeamqp-env(){      elocal- ; }
nodeamqp-usage(){
  cat << EOU
     nodeamqp-src : $(nodeamqp-src)
     nodeamqp-dir : $(nodeamqp-dir)

     Using squaremo fork for use with rabbit.js


 == node-amqp issues == 

   TypeError fixed by adding parent. as suggested by 
       http://github.com/ry/node-amqp/issues#issue/7 

/data1/env/local/env/nodejs/node-amqp/amqp.js:263
  var s = buffer.utf8Slice(buffer.read, buffer.read+length);
                 ^
/data1/env/local/env/nodejs/node-amqp/amqp.js:511
  b.utf8Write(string, b.used); // error here
    ^


   ''(svn)revert'' out of parent fix that am not very confident of, when going back to node 0.2.2

{{{
[blyth@belle7 node-amqp]$ git checkout amqp.js
[blyth@belle7 node-amqp]$ git status
}}} 



== trying node-amqp example from synopsis http://github.com/squaremo/node-amqp ==


   node synopsis.js 

   Unhandled channel error:



== forking history ... ==

  http://github.com/squaremo/node-amqp/commits/master
    * squaremo forked from ry/node-amqp at Sept 16th  2010-09-16 (the version bump to 0.0.2)  
        * this is close to node tag v0.2.2 on Sept 18th 

    * currently using node from Oct 2nd
       * http://github.com/ry/node/commit/d59512f6f406ceae4940fd9c83e8255f0c03173b








== setup/debugging on N  ==

  * export the AMQP_ private vals into env  
      nodeamqp-cfg-export 
    
  * change test/harness.js to use local config 

     var cfg = { host:process.env["AMQP_LOCAL_SERVER"], port:process.env["AMQP_LOCAL_PORT"], login:process.env["AMQP_LOCAL_USER"] , password:process.env["AMQP_LOCAL_PASSWORD"], vhost:process.env["AMQP_LOCAL_VHOST"]   };
     sys.puts( " connect to " + cfg["host"] );
     global.connection = amqp.createConnection(cfg);

 * run a single tests ...
 
     node test/test-simple.js $AMQP_LOCAL_SERVER:$AMQP_LOCAL_PORT

 * switch on debug output and collect 

     export NODE_DEBUG_AMQP=1

  [blyth@belle7 node-amqp]$ node test/test-simple.js $AMQP_LOCAL_SERVER:$AMQP_LOCAL_PORT > test-simple.out 2>&1
  [blyth@belle7 node-amqp]$ node test/test-default-exchange.js $AMQP_LOCAL_SERVER:$AMQP_LOCAL_PORT > test-default-exchange.out 2>&1


0 > connectionStart {"versionMajor":8,"versionMinor":0,"serverProperties":{"product":{"0":82,"1":97,"2":98,"3":98,"4":105,"5":116,"6":77,"7":81,"length":8,"parent":{"0":0,"1":10 .... 


 * looks like some banner is being intepreted as a queue name ...   

{{{
SET bit field nowait 0x8
execute: ^A^@^A^@^@^@"^@2^@^K^Unode-default-exchange^@^@^@^@^@^@^@^@i?1/2
got frame: [1,1,34]
1 > queueDeclareOk {"queue":"\u0000\u0000\u0000\u0000i?1/2\u0007productS\u0000\u0000\u0000\bRab","messageCount":0,"consumerCount":0}
1 < queueBind {"ticket":0,"queue":"\u0000\u0000\u0000\u0000i?1/2\u0007productS\u0000\u0000\u0000\bRab","exchange":"amq.topic","routingKey":"#","nowait":false,"arguments":{}}
SET bit field nowait 0x0
1 < basicConsume {"ticket":0,"queue":"\u0000\u0000\u0000\u0000i?1/2\u0007productS\u0000\u0000\u0000\bRab","consumerTag":"node-amqp-ctag-0.33419028320349753","noLocal":false,"noAck":true,"exclusive":false,"nowait":false,"arguments":{}}
SET bit field nowait 0x2
execute: ^A^@^A^@^@^@F^@^T^@(^Ai?1/2;NOT_FOUND - no queue '^@^@^@^@i?1/2^GproductS^@^@^@^HRab' in vhost '/'^@2^@^Ti?1/2
got frame: [1,1,70]
1 > channelClose {"replyCode":404,"replyText":"\u0000\u0000i?1/2\u0007productS\u0000\u0000\u0000\bRabbitMQ\u0007versionS\u0000\u0000\u0000\u00051.7.2\bplatformS\u0000\u0000\u0000\nErl","classId":50,"methodId":20}
Unhandled channel error: ^@^@i?1/2^GproductS^@^@^@^HRabbitMQ^GversionS^@^@^@^E1.7.2^HplatformS^@^@^@
Erl
events:12
        throw arguments[1];
                       ^
Error:

}}}






EOU
}
nodeamqp-dir(){ echo $(local-base)/env/nodejs/node-amqp ; }
nodeamqp-cd(){  cd $(nodeamqp-dir); }
nodeamqp-mate(){ mate $(nodeamqp-dir) ; }
nodeamqp-get(){
   local dir=$(dirname $(nodeamqp-dir)) &&  mkdir -p $dir && cd $dir

   git clone http://github.com/squaremo/node-amqp.git
}


nodeamqp-cfg-export(){
  local msg="=== $FUNCNAME :"
  private-
  echo $msg exporting into env
  private-export AMQP_SERVER AMQP_PORT AMQP_USER AMQP_PASSWORD AMQP_VHOST
  private-export AMQP_LOCAL_SERVER AMQP_LOCAL_PORT AMQP_LOCAL_USER AMQP_LOCAL_PASSWORD AMQP_LOCAL_VHOST
  env | grep AMQP_ 

}


nodeamqp-dbg-on(){   export NODE_DEBUG_AMQP=1;  }
nodeamqp-dbg-off(){  unset NODE_DEBUG_AMQP ;    }



