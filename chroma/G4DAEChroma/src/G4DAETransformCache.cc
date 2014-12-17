#include "G4DAEChroma/G4DAETransformCache.hh"
#include "G4DAEChroma/G4DAECommon.hh"

#include "cnpy/cnpy.h"
#include <sys/stat.h> 
#include <libgen.h>

#include "G4ThreeVector.hh"
#include "G4AffineTransform.hh"

#include <CLHEP/Vector/Rotation.h>
#include "G4RotationMatrix.hh"

#include <string>
#include <iostream>
#include <iomanip>

using namespace std ; 



void G4DAETransformCache::Serialize()
{ 
    // serialize from map into byte buffers ready for persisting 
     size_t size = m_id2transform.size();
     this->Resize(size);

     for(TransformMap_t::iterator it = m_id2transform.begin(); it != m_id2transform.end(); it++)
     {
         this->AddSerial( it->first, it->second );
     }
}

void G4DAETransformCache::DeSerialize()
{
    // copy from byte buffers into the map
    m_id2transform.clear();
    size_t size = GetSize();
    for(size_t index=0 ; index < size ; ++index )
    {
        size_t key = GetKey(index);
        G4AffineTransform* tra = GetTransform(index);
        m_id2transform[key] = *tra ;  
    }
}

void G4DAETransformCache::Dump()
{
   for(TransformMap_t::iterator it = m_id2transform.begin(); it != m_id2transform.end(); it++) 
   {
       Key_t key = it->first ; 
       cout << " key " << (void*)key << endl ; 
       cout << " key " << setw(10) << key 
            << " tr " << transform_rep( it->second )
            << endl ;


       size_t index = FindKey( key );     
       G4AffineTransform* tra = FindTransform( key );     

       cout << " idx " << setw(10) << index 
            << " tr " << transform_rep( *tra ) 
            << endl ; 



   } 
}


G4DAETransformCache::G4DAETransformCache( std::size_t itemcapacity, Key_t* key, double* data) : 
             m_itemcapacity(itemcapacity), 
             m_itemsize(4*4), 
             m_keysize(1), 
             m_data(data), 
             m_key(key), 
             m_itemcount(0) 
{
    if( m_data == NULL && m_key == NULL )
    {
        Resize(itemcapacity);      
    } 
    else
    {
        m_itemcount = m_itemcapacity ; // when loading from buffers
    }
    m_metadata = new G4DAEMetadata("{}");
}

void G4DAETransformCache::AddMetadata( const char* name, Map_t& map)
{
    m_metadata->AddMap(name, map);
}



void G4DAETransformCache::Resize(std::size_t itemcapacity)
{
   m_itemcount = 0 ; 
   m_itemcapacity = itemcapacity ;  

   if(m_data) delete[] m_data ; 
   if(m_key)  delete[] m_key ; 
     
   if(m_itemcapacity > 0){
       m_data = new double[m_itemcapacity*m_itemsize] ;
       m_key  = new Key_t[m_itemcapacity] ;
   }
}


G4DAETransformCache::~G4DAETransformCache()
{
    if(m_data) delete[] m_data ; 
    if(m_key)  delete[] m_key ; 
    delete m_metadata ;
}

void G4DAETransformCache::Archive(const char* dir)
{
    if( dir == NULL )
    {
        printf("G4DAETransformCache::Archive NULL dir, skipping \n");
        return ;
    }

    this->Serialize();

    int rc = mkdirp(dir, 0777);
    printf("G4DAETransformCache::Archive mkdirp [%s] rc %d \n", dir, rc );

    const unsigned int key_shape[] = {m_itemcount};
    const unsigned int data_shape[] = {m_itemcount,4,4};

    char path[1024];
    int len ;

    len = snprintf(path, sizeof(path)-1, "%s/%s", dir, "key.npy");
    path[len] = 0;
    printf("G4DAETransformCache::Archive npy_save keys [%s]\n", path );
    cnpy::npy_save(path,m_key,key_shape,1,"w");

    len = snprintf(path, sizeof(path)-1, "%s/%s", dir, "data.npy");
    path[len] = 0;
    printf("G4DAETransformCache::Archive npy_save data [%s]\n", path );
    cnpy::npy_save(path,m_data,data_shape,3,"w");

    len = snprintf(path, sizeof(path)-1, "%s/%s", dir, "g4materials.json");
    path[len] = 0;
    printf("G4DAETransformCache::Archive PrintToFile metadata [%s]\n", path );
    m_metadata->PrintToFile(path);

}


bool G4DAETransformCache::Exists(const char* dir)
{
    char path[1024];
    int len ;
    len = snprintf(path, sizeof(path)-1, "%s/%s", dir, "key.npy");
    path[len] = 0;
 
    struct stat   buffer;   
    bool xkey  = (stat(path, &buffer) == 0);

    len = snprintf(path, sizeof(path)-1, "%s/%s", dir, "data.npy");
    path[len] = 0;
    bool xdata = (stat(path, &buffer) == 0);

    return xkey && xdata ; 
}


G4DAETransformCache* G4DAETransformCache::Load(const char* dir)
{
    char path[1024];
    int len ;

    len = snprintf(path, sizeof(path)-1, "%s/%s", dir, "key.npy");
    path[len] = 0;

#ifdef VERBOSE
    printf("G4DAETransformCache::Load [%s] \n", path ); 
#endif
    cnpy::NpyArray akey = cnpy::npy_load(path); 

    len = snprintf(path, sizeof(path)-1, "%s/%s", dir, "data.npy");
    path[len] = 0;
#ifdef VERBOSE
    printf("G4DAETransformCache::Load [%s] \n", path ); 
#endif
    cnpy::NpyArray adata = cnpy::npy_load(path); 


    assert( akey.shape.size() == 1 );
    assert( adata.shape.size() == 3 );
    assert( adata.shape[1] == 4 && adata.shape[2] == 4 );
    assert( adata.shape[0] == akey.shape[0] );

    std::size_t itemcount = adata.shape[0] ;
    Key_t* key   = reinterpret_cast<Key_t*>(akey.data);
    double* data = reinterpret_cast<double*>(adata.data);

    G4DAETransformCache* cache = new G4DAETransformCache( itemcount, key, data );  
    cache->DeSerialize(); 
    return cache ; 
}


         
std::size_t G4DAETransformCache::GetCapacity(){
    return m_itemcapacity ; 
}

std::size_t G4DAETransformCache::GetSize(){
    return m_itemcount ; 
}

Key_t G4DAETransformCache::GetKey( std::size_t index )
{
    if( index > m_itemcapacity ) return 0 ;
    Key_t* id = m_key + index*m_keysize  ;
    return id[0];
}

// return index+1 of first key matching argument, or 0 if not found
std::size_t G4DAETransformCache::FindKey( Key_t key )
{
    for(std::size_t n=0 ; n < m_itemcount ; ++n ){
       if(m_key[n] == key) return n + 1 ; 
    }
    return 0;
}

G4AffineTransform* G4DAETransformCache::FindTransform( Key_t key )
{
    std::size_t find = FindKey( key );
    return (find == 0) ? NULL : GetTransform( find - 1 );  
} 



void G4DAETransformCache::Add( Key_t key, const G4AffineTransform&  transform )
{
    m_id2transform[key] = transform ; 
}

G4AffineTransform* G4DAETransformCache::GetSensorTransform(Key_t id)
{
    return ( m_id2transform.find(id) == m_id2transform.end()) ? NULL : &m_id2transform[id] ;
}


void G4DAETransformCache::AddSerial( Key_t key, const G4AffineTransform&  transform )
{
    // writes into the buffer
    //
    
    if( m_itemcount >= m_itemcapacity )
    {
        printf("G4DAETransformCache::AddSerial too many items itemcount %zu itemcapacity %zu \n", m_itemcount, m_itemcapacity ); 
    }

    assert(m_itemcount < m_itemcapacity );

    // pointers into the buffers, prior to incrementing m_itemcount
    double* data = m_data + m_itemcount*m_itemsize ;   
    Key_t* id = m_key + m_itemcount*m_keysize ; 
    m_itemcount++ ; 

    id[0] = key ; 

/*
  External use of G4AffineTransform data 
  ======================================= 

    R(row)(col)

     col=x col=y  col=z  col=w 

    0:Rxx  1:Rxy  2:Rxz   3:0.    row=x
    4:Ryx  5:Ryy  6:Ryz   7:0.    row=y
    8:Rzx  9:Rzy 10:Rzz  11:0.    row=z
   12:Tx  13:Ty  14:Tz   15:1.    row=w

   
  * serialization indices follows row-major layout 

  * transforms adopt row-vector convention, meaning the 
    translation portion appears in last row (rather than 
    the more familiar last column of column-vector convention)

  * this means than use of the serialized arrays from numpy
    needs to pre-multiply `np.dot(vec, m)` or post-multiply
    with transpose applied `np.dot( m.T, vec)` 

*/

    G4RotationMatrix rot = transform.NetRotation(); 
    G4ThreeVector tlate = transform.NetTranslation(); 

    data[0] = rot.xx();
    data[1] = rot.xy();
    data[2] = rot.xz();
    data[3] = 0. ;

    data[4] = rot.yx();
    data[5] = rot.yy();
    data[6] = rot.yz();
    data[7] = 0. ;

    data[8] = rot.zx();
    data[9] = rot.zy();
    data[10] = rot.zz();
    data[11] = 0. ;

    data[12] = tlate.x();
    data[13] = tlate.y();
    data[14] = tlate.z();
    data[15] = 1. ;

}


G4AffineTransform* G4DAETransformCache::GetTransform( std::size_t index )
{
    // creates the G4AffineTransform direct from the serialized buffer
    if( index > m_itemcapacity ) return NULL ;
    double* data = m_data + index*m_itemsize ;   

    CLHEP::HepRep3x3 r33( 
            data[0], data[1], data[2] ,         
            data[4], data[5], data[6] ,         
            data[8], data[9], data[10] 
    );         
    G4RotationMatrix rot(r33) ; 
    G4ThreeVector tlate(data[12],data[13],data[14]) ; 

    G4AffineTransform* tr0 = new G4AffineTransform();
    tr0->SetNetRotation(rot);
    tr0->SetNetTranslation(tlate);

    G4AffineTransform* tr1 = new G4AffineTransform(rot,tlate);
    assert( *tr0 == *tr1 );

    return tr1; 
}


