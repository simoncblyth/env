#ifndef ZMQROOT_H 
#define ZMQROOT_H

class TObject ; 

class ZMQRoot
{

public:
  ZMQRoot(const char* envvar);
  virtual ~ZMQRoot();

  void SendObject(TObject* obj);
  TObject* ReceiveObject();

  
private:
  void* fContext ;  
  void* fRequester ;  

};

#endif

