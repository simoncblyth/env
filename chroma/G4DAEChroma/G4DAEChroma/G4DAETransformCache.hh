#ifndef G4DAETRANSFORMCACHE_H
#define G4DAETRANSFORMCACHE_H
/*
G4DAETransformCache
=====================

* Holds a map of G4AffineTransforms keyed on an identifier (eg PmtId)
* Can persist/load the map to/from binary CNPY formatted key.npy 
  and data.npy files. These files can be loaded directly into python with::

       a = np.load("data.npy")

See also `env/geant4/geometry/collada/transform_cache.py` 


*/

#include "G4DAEChroma/G4DAEMetadata.hh"
#include "G4DAEChroma/G4DAEMap.hh"
#include "G4AffineTransform.hh" 

#include <map>
#include <cstddef>
#include <stdint.h>

typedef uint32_t Key_t ;
typedef std::map<Key_t,G4AffineTransform> TransformMap_t ;

class G4DAETransformCache {

public:
    G4DAETransformCache( std::size_t itemcapacity = 0, Key_t* key = NULL, double* data = NULL);
    virtual ~G4DAETransformCache();

public:
    static G4DAETransformCache* Load(const char* dir);
    static bool Exists(const char* dir);
    void Dump();

    void Archive(const char* dir);

    void Add( Key_t key, const G4AffineTransform&  transform );
    G4AffineTransform* GetSensorTransform(Key_t id);

    std::size_t GetSize();
    Key_t GetKey( std::size_t index );                     // key for index
    std::size_t FindKey( Key_t key );                      // index for key 
    G4AffineTransform* GetTransform( std::size_t index );  // transform for index
    G4AffineTransform* FindTransform( Key_t key );         // transform for key 

public:
    void AddMetadata( const char* name, Map_t& map);

protected:
    // lower level operations on the serialized bytes 

    void Serialize();
    void DeSerialize();
    std::size_t GetCapacity();
    void Resize(std::size_t itemcapacity );
    void AddSerial( Key_t key, const G4AffineTransform&  transform );

private:
    TransformMap_t m_id2transform ; 

private:
    G4DAEMetadata* m_metadata ; 

    std::size_t   m_itemcapacity ; 
    std::size_t   m_itemsize ; 
    std::size_t   m_keysize ; 
    double*       m_data ; 
    Key_t*        m_key ; 
    std::size_t   m_itemcount ; 


};

#endif

