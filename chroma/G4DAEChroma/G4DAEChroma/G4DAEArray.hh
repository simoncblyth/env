#ifndef G4DAEARRAY_H
#define G4DAEARRAY_H

#include <string>
#include <vector>

/*
  Before transport with::

     `G4DAESocket<G4DAEArray>::SendObject(G4DAEArray* a)`` 

  currently need to, with SaveToBuffer:
  
   #. create the NPY header into m_buffer
   #. copy the m_data into m_buffer + header_offset 

  This copying could be avoided by directly allocating
  the buffer with a reserve for the header, coupled with
  some header padding. 
*/


class G4DAEBuffer ;

#include "G4DAEChroma/G4DAESerializable.hh"

class G4DAEArray : public G4DAESerializable {

public:
    G4DAEArray* Create(char* bytes, size_t size);
    G4DAEArray(char* bytes, size_t size);
    G4DAEArray( std::size_t itemcapacity = 0, std::string itemshapestr = "", float* data = NULL);
    virtual ~G4DAEArray();

    void Populate( std::size_t itemcapacity, std::string itemshapestr, float* data);
    virtual void Print() const ;
    virtual void Zero();
    virtual void ClearAll();

    static std::size_t FormItemSize(const std::vector<int>& itemshape, int from=0);
    static std::string FormItemShapeString(const std::vector<int>& itemshape, int from=0);

public:
    // fulfil Serializable protocol 
    virtual void Populate( char* bytes, size_t size );
    virtual void SaveToBuffer();
    virtual const char* GetBufferBytes();
    virtual std::size_t GetBufferSize();
    virtual void DumpBuffer();

public:
    //  serialization/deserialization to file
    virtual void Save(const char* evt, const char* key, const char* tmpl );
    static G4DAEArray* Load(const char* evt, const char* key, const char* tmpl );
    static std::string GetPath( const char* evt, const char* tmpl );   

public:
   //  serialization/deserialization to NPY buffer, ready for transport over eg ZMQ
   // informal G4DAESocket protocol methods that allowing G4DAESocket<G4DAEArray> arrsock ; 
   static G4DAEArray* LoadFromBuffer(const char* buffer, std::size_t buflen);

   G4DAEBuffer* GetBuffer() const;


public:
    std::size_t GetSize() const;
    std::size_t GetItemSize() const;
    std::size_t GetCapacity() const;
    std::size_t GetBytesUsed() const;
    std::size_t GetBytes() const;
    std::string GetDigest() const; 
    std::string GetItemShapeString() const;

protected:
    // equivalent of ndarray type info
    std::vector<int> m_itemshape ; 
    std::size_t      m_itemsize ; 

protected:
    std::size_t      m_itemcount ; 
    std::size_t      m_itemcapacity ; 
    float*           m_data ; 

private:
    G4DAEBuffer*     m_buffer ; 


};

#endif


