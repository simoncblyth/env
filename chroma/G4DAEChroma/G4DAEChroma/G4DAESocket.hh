#ifndef G4DAESOCKET_H 
#define G4DAESOCKET_H

#include "G4DAEChroma/G4DAECommon.hh"

#include <cstdlib>
#include <assert.h>
#include <stdlib.h>
#include <iostream>

#include <unistd.h>


#ifdef WITH_ZMQ
#include <zmq.h>
#endif


#ifdef WITH_ZMQ
// Receive 0MQ string from socket and convert into C string
static char* s_recv (void *socket) 
{
    zmq_msg_t message;
    zmq_msg_init (&message);
    int size = zmq_msg_recv (&message, socket, 0); 
    if (size == -1) return NULL;
    char* str  = (char*)malloc(size + 1);
    memcpy (str, zmq_msg_data (&message), size); zmq_msg_close (&message);
    str [size] = 0;
    return (str);
}


// Convert C string to 0MQ string and send to socket
static int s_send (void *socket, char *str) 
{
    zmq_msg_t message;
    zmq_msg_init_size (&message, strlen(str));
    memcpy (zmq_msg_data (&message), str, strlen(str)); 
    int size = zmq_msg_send (&message, socket, 0); 
    zmq_msg_close (&message);
    return (size);
}


#endif



template <class T> 
class G4DAESocket
{
public:
  G4DAESocket(const char* envvar, const char mode='Q');
  virtual ~G4DAESocket();
 
public:
  void SendString(char* msg);
  char* ReceiveString();
  void MirrorString();

public:
  void SendObject(T* obj) const;
  T* ReceiveObject() const ;
  void MirrorObject() const;
  
private:
  void* m_context ;   
  void* m_socket ;   
  char  m_mode ;  

};



template <typename T>
G4DAESocket<T>::G4DAESocket(const char* envvar, const char mode) : m_context(NULL), m_socket(NULL), m_mode(mode)
{
#ifdef WITH_ZMQ
  char* config = getenv(envvar) ;
  if( config == NULL )
  { 
     printf( "G4DAESocket::G4DAESocket MISSING envvar [%s][%c] not configured : CANNOT SEND/RECV \n", envvar,mode );   
     return ;
  } 

  m_context = zmq_ctx_new();
  switch (m_mode)
  {
     case 'Q': 
              m_socket = zmq_socket (m_context, ZMQ_REQ);
              break ;
     case 'P': 
              m_socket = zmq_socket (m_context, ZMQ_REP);
              break ;
     default:
              m_socket = NULL ;
  }

  int rc = zmq_connect (m_socket, config );
  if(rc != 0){
      printf("G4DAESocket::G4DAESocket failed to connect using envvar %s config %s mode %c \n", envvar, config, mode); 
  }

#else
  printf( "G4DAESocket::G4DAESocket need to compile -DWITH_ZMQ and have ZMQ external \n");   
#endif
}

template <typename T>
G4DAESocket<T>::~G4DAESocket()
{
#ifdef WITH_ZMQ
   zmq_close( m_socket );
   zmq_ctx_destroy( m_context );
#endif
}





template <typename T>
void G4DAESocket<T>::SendObject(T* obj) const
{
#ifdef WITH_ZMQ
   int rc ; 
   assert( obj );

   obj->SaveToBuffer();
   const char* buffer = obj->GetBuffer();
   size_t buflen = obj->GetBufferSize();

   printf("G4DAESocket<T>::SendObject %lu \n", buflen );
   DumpBuffer( buffer, buflen );

   zmq_msg_t zmsg;
   rc = zmq_msg_init_size (&zmsg, buflen);
   assert (rc == 0);
   
   memcpy(zmq_msg_data (&zmsg), buffer, buflen );   // TODO : check for zero copy approaches

   rc = zmq_msg_send (&zmsg, m_socket, 0);
   
   if (rc == -1) {
       int err = zmq_errno();
       printf ("Error occurred during zmq_msg_send : %s\n", zmq_strerror(err));
       abort (); 
   }
  
   zmq_msg_close (&zmsg); 

#else
    printf( "G4DAESocket::SendBuffer need to compile -DWITH_ZMQ and have ZMQ external \n");   
#endif
}

template <typename T>
T* G4DAESocket<T>::ReceiveObject() const 
{
    T* object = NULL ; 
#ifdef WITH_ZMQ

    zmq_msg_t msg;

    int rc = zmq_msg_init (&msg); 
    assert (rc == 0);

    rc = zmq_msg_recv (&msg, m_socket, 0);   

    if(rc)
    {
       int err = zmq_errno();
       printf( "Error on zmq_msg_recv : %s \n ", zmq_strerror(err)) ;
    }

    assert (rc != -1);

    size_t size = zmq_msg_size(&msg); 
    void*  data = zmq_msg_data(&msg) ;

    printf("G4DAESocket<T>::ReceiveObject %zu \n", size );


    object  = T::LoadFromBuffer( reinterpret_cast<const char*>(data), size );  

    printf("G4DAESocket::ReceiveBuffer received bytes: %zu \n", size );   
    zmq_msg_close (&msg);  

#else
    printf( "G4DAESocket::ReceiveBuffer need to compile -DWITH_ZMQ and have ZMQ external \n");   
#endif

    return object ; 
}



template <typename T>
void G4DAESocket<T>::MirrorObject() const
{
#ifdef WITH_ZMQ
    assert( m_mode == 'P' );

    while(1)
    {
        zmq_msg_t request;
        int rc = zmq_msg_init(&request); assert (rc == 0);

        printf("waiting on zmq_msg_recv \n");
        rc = zmq_msg_recv (&request, m_socket,  0);

        if(rc)
        {
           int err = zmq_errno();
           printf( "Error on zmq_msg_recv : %s \n ", zmq_strerror(err)) ;
        }
        else
        {
           printf("OK zmq_msg_recv \n");
        }
        assert (rc != -1);

        sleep(1);

        zmq_msg_t reply;
        zmq_msg_init(&reply );
        zmq_msg_copy(&reply, &request);  // copy request to reply 
        zmq_msg_send (&reply, m_socket, 0);

        zmq_msg_close (&request);
        zmq_msg_close (&reply);
    }


#else
    printf( "G4DAESocket::Responder need to compile -DWITH_ZMQ and have ZMQ external \n");   
#endif
}




template <typename T>
void G4DAESocket<T>::MirrorString()
{
    while(1){

        char* req = s_recv( m_socket );
        printf("MirrorString req %s \n", req );
        free(req);        

        sleep(1);

        const char* rep = "mirror" ; 
        int size = s_send( m_socket, (char*)rep );
        printf("MirrorString size %d \n", size );

    }
}

template <typename T>
void G4DAESocket<T>::SendString(char* msg) 
{
    int size = s_send( m_socket, msg );
    printf("SendString %s size %d \n", msg, size );
}


template <typename T>
char* G4DAESocket<T>::ReceiveString() 
{
    char* msg = s_recv( m_socket );
    printf("ReceiveString %s \n", msg );
    return msg ;  
}






#endif

