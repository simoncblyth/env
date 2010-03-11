
  == Overview ==

  librootmq provides an interface to the usage of the rabbitmq-c 
  AMQP producer/consumer from root (pyROOT/cint/compiled)
  allowing TObjects to be sent and received over the network 
  via a RabbitMQ server 

  The use of a separate message queue server decouples the production and 
  consumption of TObjects, allowing asynchronous communication
  (analogous to email) between remote/local processes that perhaps are using 
  different languages : cint / pyROOT / compiled ROOT


  == Index ==    

     Makefile
         building and testing of
             1) short mains : mq_sendstring.cc 
             2) rootcint scripts : tests/test_rootsendstring.C ...
             3) pyroot scripts

         note that environment control for library access is done in this Makefile making it the hub of 
         all rootmq building testing and usage 

     pmq.py
           hook up ROOT signal/slot mechanism with the ROOT.gMQ singleton that resides in a
           separate monitor thread ... allowing non-blocking response to messages
           HOWEVER it needs to idle in a checking loop making this not a good architecture, as
           cannot use interactive python while waiting for events or update a GUI etc.. 
           
     evmq.py 
           timer based alternative that avoids the nasty polling    


     src/
            rootmq.c
                 building on the rabbitmq-c examples to provide simple C interface to AMQP/rabbitmq-c
                 functionality and integrating configuration access via my private framework  

            rootmq_collection.c
                  glib hash table of routing keys associated with dequeues of messages of limited size ...
                  threadsafe functions to get/set the messages in the queues
                  are provided ... NB messages are passed into/from these functions, ie it has 
                  little dependency on rootmq specifics  

            MQ.cc
                 rootcintable C++ interface to the rootmq standard C interface to AMQP/rabbitmq-c 
                 functionality adds support for 
                     1) conversions between TObjects and AMQP messages
                     2) simple TObject conversion into cJSON allowing communication to webapp 


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

     obsolete/

                 mq_monitor
                       monitor is started in separate thread preventing blocking, the monitor
                       thread is kept as simple as possible ... avoid use of ROOT objects 
                       BUT requires a polling loop making inflexible for integration with eg event display 

                 mq_threaded
                        initial investigations of using threading , this
                        approach suffers from deadlocks as too much is done in the monitor thread 
 
                 mq_consumebutes
                       deprecated approach using MQ::Wait that blocks on rootmq_basic_consume
                       while waiting for messages

                 mq_mapfile
                       deprecated approach that receives TObjects and writes into TMapFile 
                       for consumption by a separate process 



   == INSTALLS ==

        G        ok
        C        ok
        N        ok
        C2       ok
        P        ok

        G1       same node as P so no point ?                
        H        too old to bother trying
        H1       off limits 


   == PROBLEMS ON P ==

     a) old pcre, forced source installation and propagation of include dirs and libs into 
        build and usage commands

     b) no mercurial , and none in the yum repo ... has to pip install

     c) automake/autoconf incompatible versions 

       configure.ac:5: `automake requires `AM_CONFIG_HEADER', not `AC_CONFIG_HEADER'
            http://momentarypause.blogspot.com/2006/02/acconfigheaders-vs-amconfigheaders.html

        initially :

       [dayabaysoft@grid1 rabbitmq-c]$ automake --version
       automake (GNU automake) 1.6.3
       [dayabaysoft@grid1 rabbitmq-c]$ autoconf --version
       autoconf (GNU Autoconf) 2.57

       updating to automake17 succeeded to fix this ...
            sudo yum install automake17   

       see foot of page for details  


   == PRE-REQUISITES ==

        libprivate  ( cd ~/e/priv ; make )

        rabbitmq-c
             install with :
               rabbitmq-
               rabbitmq-c-build        (requires mercurial and pip)

               Caution the bash function does some kludging around to get the build to succeed ;
                  http://dayabay.phys.ntu.edu.tw/tracs/env/wiki/RabbitMQ#rabbitmq-cbuildonC

       cjson 
              normally just :
                    cjson- ; cjson-build 

               PATH=/usr/bin:$PATH cjson-build   
                     path shuffle to get old svn that supports SSL
                     requires root (root-;root-get;root-build)


   == PRE-REQUISITES FOR TESTING ==

        libAbtDataModel  
                 svn up -r 513 ~/a/DataModel
                 cd ~/a/DataModel
                 make
                  
                 ABERDEEN_HOME envvar defined, use aberdeen-


   == POSSIBLE ISSUES ==

      0) rabbitmq-c-preq
               " sudo pip install simplejson"

      1)  While installing rabbitmq-c with rabbitmq-c-build the build of the dependency 
          rabbitmq-codegen may require python 2.5 ?
      
make[2]: Entering directory
`/data1/env/local/env/messaging/rabbitmq-c/librabbitmq'
PYTHONPATH=/data1/env/local/env/messaging/rabbitmq-codegen python2.5
./codegen.py header
/data1/env/local/env/messaging/rabbitmq-codegen/amqp-0.8.json amqp_framing.h
/bin/sh: python2.5: command not found
make[2]: *** [amqp_framing.h] Error 127
make[2]: Leaving directory
`/data1/env/local/env/messaging/rabbitmq-c/librabbitmq'
make[1]: *** [all-recursive] Error 1
make[1]: Leaving directory `/data1/env/local/env/messaging/rabbitmq-c'
make: *** [all] Error 2

         workaround with :
              make PYTHON=python
   

      2) permission denied from SELinux in enforcing mode  on attempting to
run the test, eg "make test_sendstring"


DYLD_LIBRARY_PATH=/data1/env/local/env/messaging/rabbitmq-c/librabbitmq/.libs:/data1/env/local/env/home/priv/lib:/data1/env/local/env/messaging/cjson/lib:lib:
LD_LIBRARY_PATH=/data1/env/local/env/messaging/rabbitmq-c/librabbitmq/.libs:/data1/env/local/env/home/priv/lib:/data1/env/local/env/messaging/cjson/lib:lib:/data1/env/local/root/root_v5.21.04.source/root/lib:/cern/pro/lib:
./lib/mq_sendstring 
./lib/mq_sendstring: error while loading shared libraries: lib/librootmq.so:
cannot restore segment prot after reloc: Permission denied
make: *** [test_sendstring] Error 127


      3) running tests appear to connect to the rabbitmq broker OK , but
         messages do not show up on other consumers .... probably you forgot to
         configure the client in the file pointed to by ENV_PRIVATE_PATH   

local NOTIFYMQ_EXCHANGE=fanout.exchange
local NOTIFYMQ_EXCHANGETYPE=fanout
local NOTIFYMQ_QUEUE=belle7.nuu.edu.tw


     4) no mercurial , and package manager does not have it, eg if  "sudo yum install mercurial" 
        fails to find

        use pip or easy_install to get mercurial :
                  (sudo) pip install mercurial
                  


     5) 
         problems installling rabbitmq-c due to autoconf/automake mismatched versions
         FIXED by update of automake with "sudo yum install automake17"  


           http://autotoolset.sourceforge.net/

[dayabaysoft@grid1 rabbitmq-c]$ automake --version
automake (GNU automake) 1.6.3

[dayabaysoft@grid1 rabbitmq-c]$ sudo yum search automake
vailable package: automake17.noarch 0:1.7.8-1 from slc3-base matches with
 automake17
1 results returned
Looking in installed packages for a providing package
Installed package: automake14.noarch 0:1.4p6-6 matches with
 automake14
Installed package: automake15.noarch 0:1.5-7 matches with
 automake15
Installed package: automake.noarch 0:1.6.3-5 matches with
 automake
3 results returned



[dayabaysoft@grid1 rabbitmq-c]$ autoconf --version
autoconf (GNU Autoconf) 2.57

[dayabaysoft@grid1 rabbitmq-c]$ sudo yum search autoconf
Looking in installed packages for a providing package
Installed package: autoconf213.noarch 0:2.13-6 matches with
 autoconf213
Installed package: autoconf.noarch 0:2.57-3 matches with
 autoconf
2 results returned


[dayabaysoft@grid1 rabbitmq-c]$ automake-1.7 --version
automake (GNU automake) 1.7.8


     6)  Multiple roots ...
Processing tests/test_root2message.C...
Error in <TUnixSystem::Load>: version mismatch,
/data/env/local/env/home/rootmq/lib/librootmq.so = 52200, ROOT = 52104


    7)
Syntax error /usr/local/env/messaging/rabbitmq-c/librabbitmq/amqp.h:167:
Warning: Error occurred during reading source files
Warning: Error occurred during dictionary source generation
!!!Removing dict/rootmqDict.cxx dict/rootmqDict.h !!!

//extern int amqp_table_entry_cmp(void const *entry1, void const *entry2);
extern int amqp_table_entry_cmp(const void*entry1, const void*entry2);

  rootcint chokes on "void const*" but "const void*" (which has the same meaning) is ok 



