# === func-gen- : nodejs/socketio fgp nodejs/socketio.bash fgn socketio fgh nodejs
socketio-src(){      echo nodejs/socketio.bash ; }
socketio-source(){   echo ${BASH_SOURCE:-$(env-home)/$(socketio-src)} ; }
socketio-vi(){       vi $(socketio-source) ; }
socketio-env(){      elocal- ; }
socketio-usage(){
  cat << EOU
     socketio-src : $(socketio-src)
     socketio-dir : $(socketio-dir)

    http://github.com/LearnBoost/Socket.IO-node
    http://github.com/LearnBoost/Socket.IO-node#readme
    http://github.com/learnboost/socket.io

    Caution 2 issue trackers ...
          http://github.com/LearnBoost/Socket.IO-node/issues
          http://github.com/learnboost/socket.io/issues


     * may need to upgrade git {{{sudo yum --enablerepo=epel install git}}} 
      as requires submodule/recursive support (since 1.5.3)

       * on N upgraded git version 1.5.2.1 ==> git version 1.5.5.6
         but still cannot do "git clone ... --recursive" ... no recursive option 

       * follow https://git.wiki.kernel.org/index.php/GitSubmoduleTutorial 

       * from http://longair.net/blog/2010/06/02/git-submodules-explained/
         --recursive is 1.6.5+


[blyth@belle7 socket.io-node]$ git submodule status
-2ea263d1b64d318edeed4abe45a0f4ebae80bbff support/expresso
-bf4f9d758a222c07ac9c55aa6d3bfb7e531cf702 support/node-websocket-client
-336331a32ce7e820da2ce0a2894bde61f0666462 support/socket.io-client

[blyth@belle7 socket.io-node]$ git submodule init
Submodule 'support/expresso' (git://github.com/visionmedia/expresso.git) registered for path 'support/expresso'
Submodule 'support/node-websocket-client' (git://github.com/pgriess/node-websocket-client.git) registered for path 'support/node-websocket-client'
Submodule 'support/socket.io-client' (git://github.com/LearnBoost/Socket.IO.git) registered for path 'support/socket.io-client'

[blyth@belle7 socket.io-node]$ git config -l
[blyth@belle7 socket.io-node]$ git submodule update




  == running example ==
     
   Provide access to me ...
        IPTABLES_PORT=8080 iptables-webopen-ip <my-ip>

  === without sudo === 

    * get warning re flash transport not working (as a flash XML policy file has to be served on privileged port 843)
    * with firefox 3.6.10 /chat.html stays "Connecting..." 
    * with safari 5.0.1 succeed to chat between windows (from server console is using "websocket" transport)

 === with sudo ===

{{{
 sudo `which node` server.js
}}}

   * no warning, safari continues to work, firefox 3.6.10 still fails ... staying at "Connecting..."
     so apparently the flash fallback transport is not working   

   * after opening 843 ... firefox 3.6.10 still fails to connect 
{{{
IPTABLES_PORT=843  iptables-webopen-ip <my-ip> 
}}}

=== exclude "flashsocket" gets firefox 3.6.10 working ===

  * http://github.com/LearnBoost/Socket.IO-node/issues/#issue/63
  * clientside in chat.html
-      var socket = new io.Socket(null, {port: 8080});
+      var socket = new io.Socket(null, {port: 8080, transports:['websocket', 'server-events', 'htmlfile', 'xhr-multipart', 'xhr-polling']});

  * serverside in server.js
-var io = io.listen(server),
+var io = io.listen(server, {transports: ['websocket', 'server-events', 'htmlfile', 'xhr-multipart', 'xhr-polling']}), 

  * now, firefox 3.6.10 joins the chat ... using xhr-multipart transport 




EOU
}
socketio-dir(){ echo $(local-base)/env/nodejs/socket.io-node ; }
socketio-cd(){  cd $(socketio-dir)/$1 ; }
socketio-mate(){ mate $(socketio-dir) ; }
socketio-get(){
   local dir=$(dirname $(socketio-dir)) &&  mkdir -p $dir && cd $dir

   git clone git://github.com/LearnBoost/Socket.IO-node.git socket.io-node 

   cd socket.io-node
   git submodule init
   git submodule update

}

socketio-example(){
   socketio-cd example
   nodejs-
   nodejs-path

   node server.js
   #sudo `which node` server.js

}



