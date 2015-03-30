# === func-gen- : graphics/photonio/photonio fgp graphics/photonio/photonio.bash fgn photonio fgh graphics/photonio
photonio-src(){      echo graphics/photonio/photonio.bash ; }
photonio-source(){   echo ${BASH_SOURCE:-$(env-home)/$(photonio-src)} ; }
photonio-vi(){       vi $(photonio-source) ; }
photonio-env(){      elocal- ; }
photonio-usage(){ cat << EOU

Photonio
==========

* https://code.google.com/p/photonio/

Interested in this as example of use of 
boost::asio UDP together with GLFW.
Also uses TUIO

* http://www.tuio.org/?software

udp_server eventQueue
--------------------------

Note use of event queue with boost::mutex::scoped_lock 
in boost::asio udp_server 

/usr/local/env/graphics/photonio/src/asio.cpp::

     17 void udp_server::handle_receive(const boost::system::error_code& error,
     18                                 std::size_t size /*bytes_transferred*/) {
     19     //lock so no thread problems
     20     boost::mutex::scoped_lock lock(*io_mutex);
     21     //push the message into the event queue
     22     keimote::PhoneEvent tempEvent;
     23 
     24     tempEvent.ParseFromArray(&recv_buffer_[0], size);
     25     queue->push(tempEvent);
     26     recv_buffer_.empty();
     27     //unlock mutex on EventQueue
     28     lock.unlock();
     ..







EOU
}
photonio-dir(){ echo $(local-base)/env/graphics/photonio ; }
photonio-cd(){  cd $(photonio-dir); }
photonio-mate(){ mate $(photonio-dir) ; }
photonio-get(){
   local dir=$(dirname $(photonio-dir)) &&  mkdir -p $dir && cd $dir

    git clone https://code.google.com/p/photonio/

}

photonio-docs(){ open $(photonio-dir)/docs/html/annotated.html ; }
