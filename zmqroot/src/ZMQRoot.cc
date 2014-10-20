#include "ZMQRoot/ZMQRoot.hh"
#include "ZMQRoot/MyTMessage.hh"

#include "TMessage.h"
#include <assert.h>
#include <stdlib.h>
#include <iostream>

#ifdef WITH_ZMQ
#include <zmq.h>
#endif


ZMQRoot::ZMQRoot(const char* envvar, const char mode) : fContext(NULL), fSocket(NULL) 
{

#ifdef WITH_ZMQ
  char* config = getenv(envvar) ;
  printf( "ZMQRoot::ZMQRoot envvar [%s][%c] config [%s] \n", envvar,mode, config );   
  assert( config != NULL );

  fContext = zmq_ctx_new ();

  switch (mode)
  {
     case 'Q': 
              fSocket = zmq_socket (fContext, ZMQ_REQ);
              break ;
     case 'P': 
              fSocket = zmq_socket (fContext, ZMQ_REP);
              break ;
     default:
              fSocket = NULL ;
  }

  int rc = zmq_connect (fSocket, config );
  //assert( rc == 0); 
  if(rc != 0){
      printf("ZMQRoot::ZMQRoot failed to connect using envvar %s config %s mode %c \n", envvar, config, mode); 
  }

#else
  printf( "ZMQRoot::ZMQRoot need to compile -DWITH_ZMQ and have ZMQ external \n");   
#endif

}

void ZMQRoot::SendObject(TObject* obj)
{
   if(!obj) return ; 

#ifdef WITH_ZMQ
   assert( fSocket != NULL );

   TMessage tmsg(kMESS_OBJECT);
   tmsg.WriteObject(obj);

   char *buf     = tmsg.Buffer();
   int   bufLen = tmsg.Length();  

   zmq_msg_t zmsg;
   int rc ; 
   rc = zmq_msg_init_size (&zmsg, bufLen);
   assert (rc == 0);
   
   memcpy(zmq_msg_data (&zmsg), buf, bufLen );   // TODO : check for zero copy approaches

   rc = zmq_msg_send (&zmsg, fSocket, 0);
   
   if (rc == -1) {
       int err = zmq_errno();
       printf ("Error occurred during zmq_msg_send : %s\n", zmq_strerror(err));
       abort (); 
   }
  
   zmq_msg_close (&zmsg); 

   printf( "ZMQRoot::SendObject sent bytes: %d \n", rc );   


#else
  printf( "ZMQRoot::SendObject need to compile -DWITH_ZMQ and have ZMQ external \n");   
#endif

}


TObject* ZMQRoot::ReceiveObject()
{
    TObject* obj = NULL ;

#ifdef WITH_ZMQ

    zmq_msg_t msg;

    int rc = zmq_msg_init (&msg); 
    assert (rc == 0);

    rc = zmq_msg_recv (&msg, fSocket, 0);   
    std::cout << __LINE__ << ": " << rc << std::endl;
    std::cout << __LINE__ << ": " << zmq_strerror(errno) << std::endl;
    assert (rc != -1);

    size_t size = zmq_msg_size(&msg); 
    void* data = zmq_msg_data(&msg) ;

    printf("ZMQRoot::ReceiveObject received bytes: %zu \n", size );   

    
   // looks like leaking a MyTMessage, but doing on stack gives 
   // malloc: ... pointer being freed was not allocated
   // at the end of this scope
   //
   MyTMessage* tmsg = new MyTMessage( data , size ); 

    printf("ZMQRoot::ReceiveObject reading TObject from the TMessage \n");   
    obj = tmsg->MyReadObject(); 

    zmq_msg_close (&msg);  

#else
    printf( "ZMQRoot::SendObject need to compile -DWITH_ZMQ and have ZMQ external \n");   
#endif
    
    printf("ZMQRoot::ReceiveObject returning TObject \n");   
    return obj ;
}



ZMQRoot::~ZMQRoot()
{
#ifdef WITH_ZMQ
  if(fSocket != NULL){
      zmq_close (fSocket);
  }
  if(fContext != NULL){
       zmq_ctx_destroy(fContext); 
  }

#else
  printf( "ZMQRoot::~ZMQROOT need to compile -DWITH_ZMQ and have ZMQ external \n");   
#endif
 

}

