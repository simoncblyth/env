#include "G4DAEChroma/G4DAESocketBase.hh"
#include "G4DAEChroma/G4DAECommon.hh"
#include "G4DAEChroma/G4DAESerializable.hh"
#include "G4DAEChroma/G4DAEMetadata.hh"

#include <cstdlib>
#include <assert.h>
#include <stdlib.h>
#include <iostream>
#include <unistd.h>
#include <string.h>


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
    G4DAESerializable* link = request->GetLink() ; // assume request linked
#ifdef WITH_ZMQ

    /*
    Although looking the same the G4DAESerializable instances are not 
    all the same need to use the appropiate instance with the corresponding 
    instande : cannot materialize a NPY array from metadata ?
    */ 

    int nsend = 0 ; 
    G4DAESerializable* frame = request ;
    while(frame)  // send multipart following links in the chain
    { 
        frame->SaveToBuffer(); // serialization of object to the buffer
        const char* bytes = frame->GetBufferBytes();
        size_t size = frame->GetBufferSize();
        int flags = frame->GetLink() == NULL ? 0 : ZMQ_SNDMORE ;  

#ifdef VERBOSE
        printf("G4DAESocketBase::SendReceiveObject : nsend %d size %lu flags %d \n", nsend, size, flags );
#endif
        b_send( m_socket, bytes, size, flags );

        // TODO: stress test to check for leaks
        frame = frame->GetLink();
        nsend++ ; 
    } 

    const char* magic = request->GetMagic();
    size_t lmagic = magic ? strlen(magic) : 0 ;
    char* peek = new char[lmagic+1]; 

    typedef std::vector<G4DAESerializable*> VS_t ;
    VS_t others ; 

    int nrecv = 0 ; 
    while(1)   // receive multipart
    { 
        zmq_msg_t msg;
        b_recv( m_socket, msg );
        size_t size = zmq_msg_size(&msg); 
        void*  data = zmq_msg_data(&msg) ;
        char* cdata = reinterpret_cast<char*>(data) ; 

        strncpy(peek, cdata, lmagic);  // peek at first few bytes
        peek[lmagic] = '\0' ;

        if(strcmp(magic, peek) == 0)  // peek suggests know format (NPY serialization, json string) 
        { 
            response  = request->CreateOther( cdata, size );  
        }
        else if(link)  // needs to be multipart request in order to handle multipart response
        {
            G4DAESerializable* other = link->CreateOther( cdata, size );
            others.push_back(other);
        }  

        zmq_msg_close (&msg);

        //printf("G4DAESocketBase::SendReceiveObject : nrecv %d size %lu \n", nrecv, size );

        int more ;
        size_t more_size = sizeof(more);
        zmq_getsockopt( m_socket, ZMQ_RCVMORE, &more, &more_size);
        if(!more) break ; 

        nrecv++ ;
    } 
    delete peek ;


    // defer to end, so part order doesnt matter
    if( response )
    {
        VS_t::iterator it = others.begin();
        while(it != others.end()) 
        {
            G4DAEMetadata* m = dynamic_cast<G4DAEMetadata*>(*it);
            if(m) response->AddLink(m) ;
            it++;
        }
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






