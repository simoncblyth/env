#ifndef ZMQROOT_H 
#define ZMQROOT_H

#include <stdlib.h>

class TObject ; 

class ZMQRoot
{

public:
  ZMQRoot(const char* envvar);
  virtual ~ZMQRoot();

  void SendObject(TObject* obj);
  TObject* ReceiveObject();

  static TObject* DeSerialize(void* data, size_t size);
  
private:
  void* fContext ;  
  void* fRequester ;  

};

#endif

