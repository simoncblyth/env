# === func-gen- : network/asiozmq/asiozmq fgp network/asiozmq/asiozmq.bash fgn asiozmq fgh network/asiozmq
asiozmq-src(){      echo network/asiozmq/asiozmq.bash ; }
asiozmq-source(){   echo ${BASH_SOURCE:-$(env-home)/$(asiozmq-src)} ; }
asiozmq-vi(){       vi $(asiozmq-source) ; }
asiozmq-env(){      elocal- ; }
asiozmq-usage(){ cat << EOU

Asio ZMQ 
=========

Providing the BOOST/ASIO interfaces for ZeroMQ.

* https://github.com/yayj/asio-zmq

* header only, so below functions build the examples
* looks to be dead 


See Also 
---------

azmq- 
  looks to be an alternative that is now 
  under the umbrella of zeromq : https://github.com/zeromq/azmq


Probably a change with boost/asio
------------------------------------

::

    epsilon:asiozmq blyth$ asiozmq-make
    -- Found Boost 1.70.0 at /usr/local/opticks_externals/boost/lib/cmake/Boost-1.70.0
    --   Requested configuration: QUIET REQUIRED COMPONENTS system
    -- Found boost_headers 1.70.0 at /usr/local/opticks_externals/boost/lib/cmake/boost_headers-1.70.0
    -- Found boost_system 1.70.0 at /usr/local/opticks_externals/boost/lib/cmake/boost_system-1.70.0
    --   libboost_system.a
    -- Adding boost_system dependencies: headers
    -- Build files have been written to: /usr/local/env/network/asiozmq/example.build
    Scanning dependencies of target taskvent
    [  4%] Building CXX object CMakeFiles/taskvent.dir/taskvent.cpp.o
    In file included from /usr/local/env/network/asiozmq/example/taskvent.cpp:8:
    In file included from /usr/local/env/network/asiozmq/example/../include/asio-zmq.hpp:1:
    /usr/local/env/network/asiozmq/example/../include/asio-zmq/helpers.hpp:5:10: fatal error: 'boost/asio/posix/stream_descriptor_service.hpp' file not found
    #include <boost/asio/posix/stream_descriptor_service.hpp>
             ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    1 error generated.




potential problem from c++11 requirement
--------------------------------------------

::

    if (${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
      set(CMAKE_CXX_FLAGS "-Wall -std=c++11 -stdlib=libc++")
    else ()
      set(CMAKE_CXX_FLAGS "-Wall -std=c++11")
    endif ()


cmake fixes to build examples
------------------------------

cmake call needs option to locate my FindZMQ.cmake::

   -DCMAKE_MODULE_PATH=$ENV_HOME/cmake/Modules

Plus a few changes to use::

    delta:example blyth$ git diff CMakeLists.txt 
    diff --git a/example/CMakeLists.txt b/example/CMakeLists.txt
    index ff23762..d3fdb33 100644
    --- a/example/CMakeLists.txt
    +++ b/example/CMakeLists.txt
    @@ -10,17 +10,23 @@ endif ()
     add_definitions(-DBOOST_ASIO_HAS_STD_CHRONO)
     
     find_package(Boost REQUIRED COMPONENTS system)
    -find_library(ZMQ_LIBRARY zmq REQUIRED)
    +
    +#find_library(ZMQ_LIBRARY zmq REQUIRED)
    +find_package(ZMQ REQUIRED)
     
     file(GLOB example_SRCS "${CMAKE_SOURCE_DIR}/*.cpp")
     
     include_directories(
         ${CMAKE_SOURCE_DIR}/../include
         ${Boost_INCLUDE_DIRS}
    +    ${ZMQ_INCLUDE_DIRS}
         )
     
    +
    +
     foreach(SRC ${example_SRCS})
       get_filename_component(EXE ${SRC} NAME_WE)
       add_executable(${EXE} ${SRC})
    -  target_link_libraries(${EXE} ${ZMQ_LIBRARY} ${Boost_LIBRARIES})
    +  #target_link_libraries(${EXE} ${ZMQ_LIBRARY} ${Boost_LIBRARIES})
    +  target_link_libraries(${EXE} ${ZMQ_LIBRARIES} ${Boost_LIBRARIES})
     endforeach()


examples
-------------


identity
~~~~~~~~~

Output perplexing until you realise are seeing
the internal multipart structure of ZMQ messages 
with an identifier then empty, then the body.

::

    delta:example.build blyth$ ./identity 
    ----------------------------------------
    ?A?

    ROUTER uses a generated UUID
    ----------------------------------------
    PEER2

    ROUTER socket uses REQ's socket identity
    delta:example.build blyth$ 


rrbroker/rrclient/rrworker
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* rrbroker

  * instantiation binds sockets and attaches async handers to invoke handle_recv

    * frontend_.bind("tcp://*:5559")   

      * rrclient REQ write_message/read_message to/from this socket)

    * backend_.bind("tcp://*:5560")    

      * rrworker REP async_read_message with handle_req which write_message back
        and then async_read_message with handle_req to keep going 

  * handle_recv

    * grabs msg into tmp via a swap
    * forwards message to the other one
    * invokes async_recv_message on the receiver to keep the ball rolling



FIXED : getting duplicate symbols with these three
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    boost::asio::error::zmq_category()
    boost::asio::error::make_error_code(boost::asio::error::zmq_error)
    std::to_string(boost::asio::zmq::frame const&)


Whats special with those ?

* symbols defined outside of class or struct definitions 
  which do not have an inline

* FIXED : by adding "inline" to those three::

    -const system::error_category& zmq_category()
    +inline const system::error_category& zmq_category()

    -system::error_code make_error_code(zmq_error e)
    +inline system::error_code make_error_code(zmq_error e)

    -std::string to_string(boost::asio::zmq::frame const& frame)
    +inline std::string to_string(boost::asio::zmq::frame const& frame)


C++ duplicate symbols at linking
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* pragma once does not help, that is to avoid duplication 
  of headers within one compilation unit, not between units

* methods defined inside the class definition are implicitly 
  inline and do not cause dyplicate symbols 

* sometimes can use extern to avoid, how ?

* as above shows adding "inline" can be an easy fix sometimes


EOU
}
asiozmq-dir(){  echo $(local-base)/env/network/asiozmq ; }
asiozmq-idir(){ echo $(asiozmq-dir)/include ; }
asiozmq-sdir(){ echo $(asiozmq-dir)/example ; }
asiozmq-bdir(){ echo $(asiozmq-dir)/example.build ; }
asiozmq-edir(){ echo $(env-home)/network/asiozmq; }

asiozmq-cd(){   cd $(asiozmq-dir) ; }
asiozmq-scd(){  cd $(asiozmq-sdir)  ; }
asiozmq-bcd(){  cd $(asiozmq-bdir) ; }
asiozmq-icd(){  cd $(asiozmq-idir)/asio-zmq ; }
asiozmq-ecd(){  cd $(asiozmq-edir) ; }

asiozmq-info(){ cat << EOI
  
   asiozmq-dir  : $(asiozmq-dir)
   asiozmq-idir : $(asiozmq-idir)
   asiozmq-sdir : $(asiozmq-sdir)
   asiozmq-bdir : $(asiozmq-bdir)

EOI
}

#asiozmq-url(){ echo https://github.com/yayj/asio-zmq.git ; }
asiozmq-url(){ echo git://github.com/simoncblyth/asio-zmq.git ; }

asiozmq-get(){
   local dir=$(dirname $(asiozmq-dir)) &&  mkdir -p $dir && cd $dir
   [ ! -d asiozmq ] && git clone $(asiozmq-url) asiozmq 
}
asiozmq-wipe(){
   local bdir=$(asiozmq-bdir)  ;
   rm -rf $bdir
}
asiozmq-cmake(){
   local iwd=$PWD
   local sdir=$(asiozmq-sdir) ;
   local bdir=$(asiozmq-bdir)  ;
   mkdir -p $bdir
   asiozmq-bcd
   cmake $sdir -DCMAKE_MODULE_PATH=$ENV_HOME/cmake/Modules
   cd $iwd
}
asiozmq-make(){
   local iwd=$PWD
   asiozmq-bcd
   make $*
   cd $iwd
}
asiozmq--(){
   asiozmq-wipe
   asiozmq-cmake
   asiozmq-make
}


