#ifndef G4DAEARRAY_H
#define G4DAEARRAY_H

#include <string>
#include <vector>

class G4DAEArray {

public:
  G4DAEArray( std::size_t itemcapacity = 0, std::string itemshapestr = "", float* data = NULL);
  virtual ~G4DAEArray();
  virtual void Print() const ;

  static std::size_t FormItemSize(const std::vector<int>& itemshape, int from=0);
  static std::string FormItemShapeString(const std::vector<int>& itemshape, int from=0);

public:
  //  serialization/deserialization to file
  virtual void Save(const char* evt, const char* key, const char* tmpl );
  static G4DAEArray* Load(const char* evt, const char* key, const char* tmpl );
  static std::string GetPath( const char* evt, const char* tmpl );   

public:
  //  serialization/deserialization to NPY buffer, ready for transport over eg ZMQ
  // informal G4DAESocket protocol methods that allowing G4DAESocket<G4DAEArray> arrsock ; 
  virtual void SaveToBuffer();
  static G4DAEArray* LoadFromBuffer(const char* buffer, std::size_t buflen);
  virtual const char* GetBuffer() const; 
  virtual std::size_t GetBufferSize() const; 
  virtual void DumpBuffer() const ;


public:
  std::size_t GetSize() const;
  std::size_t GetItemSize() const;
  std::size_t GetCapacity() const;
  std::size_t GetBytesUsed() const;
  std::size_t GetBytes() const;
  std::string GetDigest() const; 
  std::string GetItemShapeString() const;

protected:
    std::size_t      m_itemcapacity ; 
    std::size_t      m_itemcount ; 
    std::vector<int> m_itemshape ; 
    std::size_t      m_itemsize ; 
    float*           m_data ; 

private:
    char*            m_buffer ;
    std::size_t      m_buffersize ; 


};

#endif


