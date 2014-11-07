#ifndef G4DAESOCKET_H 
#define G4DAESOCKET_H

/*

G4DAESocket<T>
=================
  
Types T compatible with G4DAESocket 
must implement the G4DAESerializable protocol
methods and a static Create from bytes, size 
function.

*/

#include "G4DAEChroma/G4DAECommon.hh"
#include "G4DAEChroma/G4DAESocketBase.hh"

#ifdef WITH_ZMQ
#include <zmq.h>
#endif

template <class T> 
class G4DAESocket : public G4DAESocketBase 
{
public:
  G4DAESocket(const char* envvar, const char mode='Q') : G4DAESocketBase( envvar, mode ) {};
  virtual ~G4DAESocket(){};
 
public:
  void SendObject(T* obj) const;
  T* ReceiveObject() const ;

};


template <typename T>
void G4DAESocket<T>::SendObject(T* obj) const
{
#ifdef WITH_ZMQ
   obj->SaveToBuffer(); 
   const char* bytes = obj->GetBufferBytes();
   size_t size = obj->GetBufferSize();
   printf("G4DAESocket<T>::SendObject %lu \n", size );
   b_send( m_socket, bytes, size );
#else
   printf( "G4DAESocket<T>::SendObject : need to compile -DWITH_ZMQ and have ZMQ external \n");   
#endif
}

template <typename T>
T* G4DAESocket<T>::ReceiveObject() const 
{
    T zombie ; 
#ifdef WITH_ZMQ

    zmq_msg_t msg;
    b_recv( m_socket, msg );
    size_t size = zmq_msg_size(&msg); 
    void*  data = zmq_msg_data(&msg) ;

    T* object  = zombie.Create( reinterpret_cast<char*>(data), size );  

    zmq_msg_close (&msg);  

    printf("G4DAESocket::ReceiveObject received bytes: %zu \n", size );   
#else
    printf("G4DAESocket::ReceiveObject need to compile -DWITH_ZMQ and have ZMQ external \n");   
#endif
    return object ; 
}


#endif

