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
#include <limits.h>

class G4DAEArray : public G4DAESerializable {
public:
    static const char* MAGIC ; 
    static const size_t INITCAPACITY ; 
    static const float GROWTH ; 
public:
    G4DAEArray* CreateOther(char* bytes, size_t size);

    G4DAEArray(char* bytes, size_t size, float growth=1.5 );
    G4DAEArray( std::size_t initcapacity = INITCAPACITY , std::string itemshapestr = "", float* data = NULL, float growth=GROWTH );
    virtual ~G4DAEArray();

public:
    G4DAEArray* Slice( int start, int stop, int step );
    G4DAEArray( G4DAEArray* src, int start=INT_MAX, int stop=INT_MAX, int step=INT_MAX );
    static void Transfer(G4DAEArray* dest , G4DAEArray* src, size_t start=INT_MAX, size_t stop=INT_MAX, int step=INT_MAX ); // INT_MAX means None

protected:
    void InitializeItemShape(std::string itemshapestr);

public:
    void Allocate( size_t nitems );
    void Extend(size_t nitems );
    void Populate( std::size_t itemcapacity, std::string itemshapestr, float* data);
    virtual void Print(const char* msg="G4DAEArray::Print") const ;
    virtual void Zero();
    virtual void ClearAll();

    static size_t FormItemSize(const std::vector<int>& itemshape, size_t from=0);
    static std::string FormItemShapeString(const std::vector<int>& itemshape, size_t from=0);
public:
    // fulfil Serializable protocol 
    virtual void Populate( char* bytes, size_t size );
    virtual void SaveToBuffer();
    virtual const char* GetBufferBytes();
    virtual size_t GetBufferSize();
    virtual void DumpBuffer();
    virtual const char* GetMagic();

public:
    //  serialization/deserialization to file
    virtual void Save(const char* evt, const char* key, const char* tmpl );
    virtual void SavePath(const char* path, const char* key="NPL");
    static G4DAEArray* Load(const char* evt, const char* key, const char* tmpl );
    static G4DAEArray* LoadPath(const char* path, const char* key="NPL");
    static std::string GetPath( const char* evt, const char* tmpl );   

public:
   //  serialization/deserialization to NPY buffer, ready for transport over eg ZMQ
   // informal G4DAESocket protocol methods that allowing G4DAESocket<G4DAEArray> arrsock ; 
   static G4DAEArray* LoadFromBuffer(const char* buffer, std::size_t buflen);

   G4DAEBuffer* GetBuffer() const;

public:
    float* GetItemPointer(std::size_t index);
    float* GetNextPointer();


public:
    std::size_t GetSize() const;
    std::size_t GetItemSize() const;
    std::size_t GetCapacity() const;
    std::size_t GetInitCapacity() const;
    std::size_t GetBytesUsed() const;
    std::size_t GetBytes() const;
    std::string GetDigest() const; 
    std::string GetItemShapeString() const;
    float GetGrowthFactor() const;

protected:
    // equivalent of ndarray type info
    std::vector<int> m_itemshape ; 
    std::size_t      m_itemsize ; 
    std::size_t      m_initcapacity ; 
    float            m_growthfactor ; 

protected:
    std::size_t      m_itemcount ; 
    std::size_t      m_itemcapacity ; 
    float*           m_data ; 

private:
    G4DAEBuffer*     m_buffer ; 

public:

    // static template functions for array holder types like G4DAEHitList and G4DAEPhotonList
    template<typename T>
    static T* Load(const char* evt, const char* key=T::KEY, const char* tmpl=T::TMPL)
    {
        G4DAEArray* array = G4DAEArray::Load(evt, key, tmpl);
        return new T(array);
    }

    template<typename T>
    static T* LoadPath(const char* path, const char* key=T::KEY)
    {
        G4DAEArray* array = G4DAEArray::LoadPath(path, key);
        return new T(array);
    }

    // instance template methods just to access default arguments 
    // from the corresponding class static consts

    template<typename T>
    void Save(const char* evt, const char* key=T::KEY, const char* tmpl=T::TMPL)
    {
        this->Save(evt, key, tmpl);
    }

    template<typename T>
    void SavePath(const char* path, const char* key=T::KEY)
    {
        this->SavePath(path, key);
    }



};

#endif



