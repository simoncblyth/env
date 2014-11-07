#include "G4DAEChroma/G4DAESocketBase.hh"
#include "G4DAEChroma/G4DAECommon.hh"
#include "G4DAEChroma/G4DAESerializable.hh"

#include <cstdlib>
#include <assert.h>
#include <stdlib.h>
#include <iostream>
#include <unistd.h>


#ifdef WITH_ZMQ
#include <zmq.h>
#endif


G4DAESocketBase::G4DAESocketBase( const char* envvar , char mode ) :  m_context(NULL), m_socket(NULL), m_mode(mode)
{
#ifdef WITH_ZMQ
  char* config = getenv(envvar) ;
  if( config == NULL )
  { 
     printf( "G4DAESocketBase::G4DAESocketBase MISSING envvar [%s][%c] not configured : CANNOT SEND/RECV \n", envvar,mode );   
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
      printf("G4DAESocketBase::G4DAESocketBase failed to connect using envvar %s config %s mode %c \n", envvar, config, mode); 
  }

#else
  printf( "G4DAESocketBase::G4DAESocketBase need to compile -DWITH_ZMQ and have ZMQ external \n");   
#endif
}

G4DAESocketBase::~G4DAESocketBase()
{
#ifdef WITH_ZMQ
   zmq_close( m_socket );
   zmq_ctx_destroy( m_context );
#endif
}



void G4DAESocketBase::MirrorString()
{
    while(1){

        char* request = s_recv( m_socket );
        printf("G4DAESocketBase::MirrorString request \"%s\" \n", request );
        std::string msg(request);
        free(request);        

        sleep(1);

        char* response = const_cast<char*>(msg.c_str()); 
        int size = s_send( m_socket, response );
        printf("G4DAESocketBase::MirrorString response \"%s\" size %d \n", response, size );

    }
}

void G4DAESocketBase::SendString(char* request) 
{
    int size = s_send( m_socket, request );
    printf("G4DAESocketBase::SendString \"%s\" bytes sent %d \n", request, size );
}


char* G4DAESocketBase::ReceiveString() 
{
    char* request = s_recv( m_socket );
    printf("G4DAESocketBase::ReceiveString request \"%s\" \n", request );
    return request ;  
}


G4DAESerializable* G4DAESocketBase::SendReceiveObject(G4DAESerializable* request) const
{
    G4DAESerializable* response = NULL ; 
#ifdef WITH_ZMQ
    // send
    {
        request->SaveToBuffer(); // serialization of object to the buffer
        const char* bytes = request->GetBufferBytes();
        size_t size = request->GetBufferSize();
        printf("G4DAESocketBase::SendReceiveObject : send  %lu \n", size );
        b_send( m_socket, bytes, size );
    } 
    // receive
    { 
        zmq_msg_t msg;
        b_recv( m_socket, msg );
        size_t size = zmq_msg_size(&msg); 
        void*  data = zmq_msg_data(&msg) ;

        // using static Create method of request object to form response 
        // this allows this to work without template type crutch 

        response  = request->Create( reinterpret_cast<char*>(data), size );  
        zmq_msg_close (&msg);  
    } 
#else
    printf( "G4DAESocketBase::SendObject : need to compile -DWITH_ZMQ and have ZMQ external \n");   
#endif
    return response ; 
}



// NB no need for types, this is just shoveling bytes 
void G4DAESocketBase::MirrorObject() const
{
#ifdef WITH_ZMQ
    assert( m_mode == 'P' );

    while(1)
    {
        zmq_msg_t request;
        b_recv( m_socket, request );

        sleep(1);

        zmq_msg_t reply;
        zmq_msg_init(&reply );
        zmq_msg_copy(&reply, &request);  // copy request to reply 
        zmq_msg_send (&reply, m_socket, 0);

        zmq_msg_close (&request);
        zmq_msg_close (&reply);
    }

#else
    printf( "G4DAESocketBase::MirrorObject need to compile -DWITH_ZMQ and have ZMQ external \n");   
#endif
}






