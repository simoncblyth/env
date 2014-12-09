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


Issues
--------

portable keys
~~~~~~~~~~~~~~~~~

Argh size_t not a portable key between 32 and 64 bit 
architectures ? Or maybe cnpy is deficient, as the python load
succeeds to convert appropriately.

Big numbers are keys from the dump, truncating them down to uint32
gives PMTID values::

    In [7]: "0x%x" % np.uint32(72339077621415937) 
    Out[7]: '0x1010001'

    In [8]: "0x%x" % np.uint32(72339086211350531)
    Out[8]: '0x1010003'



*/

#include <cstddef>


//#include <cstdint>   C++0x   depends on newish C++0x
//typedef std::uint32_t Key_t ;

#include <stdint.h>
typedef uint32_t Key_t ;

#include "G4AffineTransform.hh" 
#include <map>

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
    std::size_t   m_itemcapacity ; 
    std::size_t   m_itemsize ; 
    std::size_t   m_keysize ; 
    double*       m_data ; 
    Key_t*        m_key ; 
    std::size_t   m_itemcount ; 


};

#endif

