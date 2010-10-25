# === func-gen- : nodejs/nodejs fgp nodejs/nodejs.bash fgn nodejs fgh nodejs
nodejs-src(){      echo nodejs/nodejs.bash ; }
nodejs-source(){   echo ${BASH_SOURCE:-$(env-home)/$(nodejs-src)} ; }
nodejs-vi(){       vi $(nodejs-source) ; }
nodejs-env(){      elocal- ; }



nodejs-usage(){
  cat << EOU
     nodejs-src : $(nodejs-src)
     nodejs-dir : $(nodejs-dir)

     http://nodejs.org/
     http://github.com/ry/node/tree/master 
     http://github.com/ry/node/issues

  * building runs into issue with libev 
      * follow suggestion in  http://github.com/ry/node/issues#issue/170 and checkout prior revision to proceed with build 

      * http://github.com/ry/node/commit/d59512f6f406ceae4940fd9c83e8255f0c03173b
         * back to Oct 2nd

{{{
default/deps/libeio/eio_1.o: In function `eio__sync_file_range':
/data1/env/local/env/nodejs/node/build/../deps/libeio/eio.c:874: undefined reference to `sync_file_range'
/data1/env/local/env/nodejs/node/build/../deps/libeio/eio.c:874: undefined reference to `sync_file_range'
default/deps/libev/ev_1.o: In function `ev_signal_stop':
/data1/env/local/env/nodejs/node/build/../deps/libev/ev.c:2874: undefined reference to `signalfd'
default/deps/libev/ev_1.o: In function `evpipe_init':
/data1/env/local/env/nodejs/node/build/../deps/libev/ev.c:1243: undefined reference to `eventfd'
/data1/env/local/env/nodejs/node/build/../deps/libev/ev.c:1245: undefined reference to `eventfd'
default/deps/libev/ev_1.o: In function `ev_signal_start':
/data1/env/local/env/nodejs/node/build/../deps/libev/ev.c:2813: undefined reference to `signalfd'
/data1/env/local/env/nodejs/node/build/../deps/libev/ev.c:2790: undefined reference to `signalfd'
/data1/env/local/env/nodejs/node/build/../deps/libev/ev.c:2792: undefined reference to `signalfd'
collect2: ld returned 1 exit status
Waf: Leaving directory `/data1/env/local/env/nodejs/node/build'
Build failed:  -> task failed (err #1): 
}}}


  * testing yields 2/132 fails

{{{
[blyth@belle7 node]$ make test
Waf: Entering directory `/data1/env/local/env/nodejs/build/node/build'
DEST_OS: linux
DEST_CPU: x86
Parallel Jobs: 1
Waf: Leaving directory `/data1/env/local/env/nodejs/build/node/build'
'build' finished successfully (0.082s)
python tools/test.py --mode=release simple message
=== release test-http-upgrade-client2 ===                          
Path: simple/test-http-upgrade-client2
node.js:50
    throw e;
    ^
Error: ECONNREFUSED, Connection refused
    at IOWatcher.callback (net:867:22)
    at node.js:599:9
Command: build/default/node /data1/env/local/env/nodejs/build/node/test/simple/test-http-upgrade-client2.js
=== release test-dgram-multicast ===                                    
Path: simple/test-dgram-multicast
sent 'First message to send' to 224.0.0.112346
sent 'Second message to send' to 224.0.0.112346
sent 'Third message to send' to 224.0.0.112346
sent 'Fourth message to send' to 224.0.0.112346
sendSocket closed
Command: build/default/node /data1/env/local/env/nodejs/build/node/test/simple/test-dgram-multicast.js
--- TIMEOUT ---
[01:19|% 100|+ 132|-   2]: Done                                       
make: *** [test] Error 1
}}}


== revert to v0.2.2 to correspond to forking time of  ry/node-amqp  to squaremo/node-amqp ==

    * checkout tag v0.2.2, clean, configure and build ...
{{{
[blyth@belle7 node]$ git checkout v0.2.2
HEAD is now at 7bf46bc... bump version to v0.2.2
   ...

[blyth@belle7 node-amqp]$ node -v
v0.2.2
}}}

   * one failed test 
{{{
=== release test-dgram-multicast ===                                   
Path: simple/test-dgram-multicast
sent 'First message to send' to 224.0.0.112346
sent 'Second message to send' to 224.0.0.112346
sent 'Third message to send' to 224.0.0.112346
sent 'Fourth message to send' to 224.0.0.112346
sendSocket closed
Command: build/default/node /data1/env/local/env/nodejs/build/node/test/simple/test-dgram-multicast.js
--- TIMEOUT ---
[01:19|% 100|+ 126|-   1]: Done                                       
make: *** [test] Error 1
}}}

EOU

}
nodejs-dir(){ echo $(local-base)/env/nodejs/build/node ; }
nodejs-idir(){ echo $(local-base)/env/nodejs/install ; }
nodejs-cd(){  cd $(nodejs-dir); }
nodejs-mate(){ mate $(nodejs-dir) ; }
nodejs-url(){ echo git://github.com/ry/node.git ;  } 

nodejs-path(){
  export PATH=$(nodejs-idir)/bin:$PATH
}

nodejs-wipe(){
   local dir=$(dirname $(nodejs-dir)) &&  mkdir -p $dir && cd $dir
   rm -rf node
}

nodejs-get(){
   local dir=$(dirname $(nodejs-dir)) &&  mkdir -p $dir && cd $dir
   [ ! -d "node" ] && git clone $(nodejs-url) 
   nodejs-cd
   #git checkout d59512f6f406ceae4940

   ## get into the ballpark of when squaremo forked from ry/node-amqp 
   git checkout v0.2.2
}

nodejs-build(){
  #nodejs-get
  nodejs-cd
  ./configure --prefix=$(nodejs-idir)
  make
  make install 

}







