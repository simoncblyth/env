#include "ZMQRoot.hh"

#include "TMessage.h"
#include <assert.h>

#ifdef WITH_ZMQ
#include <zmq.h>
#endif


ZMQRoot::ZMQRoot(const char* envvar) : fContext(NULL), fRequester(NULL) 
{

#ifdef WITH_ZMQ
  char* config = getenv(envvar) ;
  printf( "ZMQRoot::ZMQRoot envvar [%s] config [%s] \n", envvar, config );   
  assert( config != NULL );

  fContext = zmq_ctx_new ();
  fRequester = zmq_socket (fContext, ZMQ_REQ);

  int rc = zmq_connect (fRequester, config );
  assert( rc == 0); 
#else
  printf( "ZMQRoot::ZMQRoot need to compile -DWITH_ZMQ and have ZMQ external \n");   
#endif

}

void ZMQRoot::SendObject(TObject* obj)
{
   if(!obj) return ; 

#ifdef WITH_ZMQ
   assert( fRequester != NULL );

   TMessage* tmsg = new TMessage(kMESS_OBJECT);
   tmsg->WriteObject(obj);

   char *buf     = tmsg->Buffer();
   int   bufLen = tmsg->Length();  

   int rc ; 
   zmq_msg_t zmsg;

   rc = zmq_msg_init_size (&zmsg, bufLen);
   assert (rc == 0);
   memcpy(zmq_msg_data (&zmsg), buf, bufLen );   // TODO : check for zero copy approaches

   rc = zmq_msg_send (&zmsg, fRequester, 0);
   if (rc == -1) {
       int err = zmq_errno();
       printf ("Error occurred during zmq_msg_send : %s\n", zmq_strerror(err));
       abort (); 
   }

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

    rc = zmq_msg_recv (&msg, fRequester, 0);   
    assert (rc != -1);

    size_t size = zmq_msg_size(&msg); 
    void* data = zmq_msg_data(&msg) ;

    printf("ZMQRoot::ReceiveObject received bytes: %zu \n", size );   

    obj = DeSerialize( data, size ); 

    zmq_msg_close (&msg);

#else
  printf( "ZMQRoot::SendObject need to compile -DWITH_ZMQ and have ZMQ external \n");   
#endif
    
    return obj ;
}



ZMQRoot::~ZMQRoot()
{
#ifdef WITH_ZMQ
  if(fRequester != NULL){
      zmq_close (fRequester);
  }
  if(fContext != NULL){
       zmq_ctx_destroy(fContext); 
  }

#else
  printf( "ZMQRoot::~ZMQROOT need to compile -DWITH_ZMQ and have ZMQ external \n");   
#endif
 

}

