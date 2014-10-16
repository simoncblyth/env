#ifndef ZMQROOT_H 
#define ZMQROOT_H

class TObject ; 

class ZMQRoot
{

public:
  ZMQRoot(const char* envvar, const char mode='Q');
  virtual ~ZMQRoot();

  void SendObject(TObject* obj);
  TObject* ReceiveObject();

  
private:
  void* fContext ;  
  void* fSocket ;  

};

#endif

