#ifndef G4DAESOCKETBASE_H
#define G4DAESOCKETBASE_H

//
// see if a non-templated approach is viable
//
//   * Send is OK as have the type, 
//   * for Receive dont have the type, 
//   * maybe a combined SendReceive could use the static method of the request 
//     object type to Create the response object 
//  


class G4DAESerializable ;
//class G4DAESerializablePhotons ;


class G4DAESocketBase
{
public:
  G4DAESocketBase(const char* envvar, const char mode='Q');
  virtual ~G4DAESocketBase();
 
public:
  virtual void SendString(char* msg);
  virtual char* ReceiveString();
  virtual void MirrorString();
 
public:
  virtual void MirrorObject() const;
  virtual G4DAESerializable* SendReceiveObject(G4DAESerializable* obj) const;
  //virtual G4DAESerializablePhotons* SendReceiveObject(G4DAESerializablePhotons* obj) const;

protected:
  void* m_context ;   
  void* m_socket ;   
  char  m_mode ;  


};

#endif
