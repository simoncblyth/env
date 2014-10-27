#ifndef G4DAETRANSFORMCACHE_H
#define G4DAETRANSFORMCACHE_H

#include <cstddef>
#include "G4AffineTransform.hh" 
#include <map>

typedef std::map<std::size_t,G4AffineTransform> TransformMap_t ;

class G4DAETransformCache {

public:
    G4DAETransformCache( std::size_t itemcapacity = 0, std::size_t* key = NULL, double* data = NULL);
    virtual ~G4DAETransformCache();

public:
    static G4DAETransformCache* Load(const char* dir);
    static bool Exists(const char* dir);
    void Dump();

    void Archive(const char* dir);

    void Add( std::size_t key, const G4AffineTransform&  transform );
    G4AffineTransform* GetSensorTransform(std::size_t id);

protected:
    // lower level operations on the serialized bytes 

    void Serialize();
    void DeSerialize();

    std::size_t GetCapacity();
    std::size_t GetSize();

    void Resize(std::size_t itemcapacity );
    std::size_t GetKey( std::size_t index );
    std::size_t FindKey( std::size_t key );
    G4AffineTransform* GetTransform( std::size_t index );
    G4AffineTransform* FindTransform( std::size_t key );

    void AddSerial( std::size_t key, const G4AffineTransform&  transform );

private:
    TransformMap_t m_id2transform ; 

private:
    std::size_t   m_itemcapacity ; 
    std::size_t   m_itemsize ; 
    std::size_t   m_keysize ; 
    double*       m_data ; 
    std::size_t*  m_key ; 
    std::size_t   m_itemcount ; 


};

#endif

