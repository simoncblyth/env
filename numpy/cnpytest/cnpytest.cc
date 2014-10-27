#include "cnpy.h"

#include <cstddef>

#include "G4ThreeVector.hh"
#include "G4AffineTransform.hh"

#include <iostream>
#include <sstream>

using namespace std ; 

class TransformCache {
  public:
     TransformCache( std::size_t itemcapacity, std::size_t* key = NULL, double* data = NULL) : 
             m_itemcapacity(itemcapacity), 
             m_itemsize(4*4), 
             m_keysize(1), 
             m_data(data), 
             m_key(key), 
             m_itemcount(0) 
     {
         if( m_data == NULL && m_key == NULL )
         {
             m_data = new double[m_itemcapacity*m_itemsize] ;
             m_key  = new std::size_t[m_itemcapacity] ;
         } 
         else
         {
            // when loading from buffers
             m_itemcount = m_itemcapacity ; 
         }
     }
     virtual ~TransformCache()
     {
          delete[] m_data ; 
          delete[] m_key ; 
     }

    void Archive()
    {
        const unsigned int key_shape[] = {m_itemcount};
        cnpy::npy_save("key.npy",m_key,key_shape,1,"w");

        const unsigned int data_shape[] = {m_itemcount,4,4};
        cnpy::npy_save("data.npy",m_data,data_shape,3,"w");
    }

    static TransformCache* Load()
    {
        cnpy::NpyArray akey = cnpy::npy_load("key.npy"); 
        assert( akey.shape.size() == 1 );

        cnpy::NpyArray adata = cnpy::npy_load("data.npy"); 
        assert( adata.shape.size() == 3 );
        assert( adata.shape[1] == 4 && adata.shape[2] == 4 );
   
        assert( adata.shape[0] == akey.shape[0] );

        std::size_t itemcount = adata.shape[0] ;
        std::size_t* key = reinterpret_cast<std::size_t*>(akey.data);
        double* data = reinterpret_cast<double*>(adata.data);

        return new TransformCache( itemcount, key, data );  
    }
          
    std::size_t GetCapacity(){
        return m_itemcapacity ; 
    }

    std::size_t GetSize(){
        return m_itemcount ; 
    }

    std::size_t GetKey( std::size_t index )
    {
        if( index > m_itemcapacity ) return 0 ;
        std::size_t* id = m_key + index*m_keysize  ;
        return id[0];
    }

    // return index+1 of first key matching argument, or 0 if not found
    std::size_t FindKey( std::size_t key )
    {
        for(std::size_t n=0 ; n < m_itemcount ; ++n ){
            if(m_key[n] == key) return n + 1 ; 
        }
        return 0;
    }

    G4AffineTransform* FindTransform( std::size_t key )
    {
        std::size_t find = FindKey( key );
        return (find == 0) ? NULL : GetTransform( find - 1 );  
    } 


    G4AffineTransform* GetTransform( std::size_t index )
    {
        if( index > m_itemcapacity ) return NULL ;
        double* data = m_data + index*m_itemsize ;   

        G4Rep3x3 r33( 
            data[0], data[1], data[2] ,         
            data[4], data[5], data[6] ,         
            data[8], data[9], data[10] 
        );         
        G4RotationMatrix rot(r33) ; 
        G4ThreeVector tlate(data[3],data[7],data[11]) ; 

        G4AffineTransform* transform = new G4AffineTransform();
        transform->SetNetRotation(rot);
        transform->SetNetTranslation(tlate);
        return transform; 
    }


    void Add( std::size_t key, const G4AffineTransform&  transform )
    {
           assert(m_itemcount < m_itemcapacity );

           double* data = m_data + m_itemcount*m_itemsize ;   
           std::size_t* id = m_key + m_itemcount*m_keysize ; 

           m_itemcount++ ; 

           id[0] = key ; 

/*

  Indexed access to G4AffineTransform follows
  an unusual convention, with translation
  along the bottom ... so not using  that for clarity 

    0:Rxx  1:Rxy  2:Rxz   3:0.
    4:Ryx  5:Ryy  6:Ryz   7:0.  
    8:Rzx  9:Rzy 10:Rzz  11:0.
   12:Tx  13:Ty  14:Tz   15:1.

*/

           G4RotationMatrix rot = transform.NetRotation(); 
           G4ThreeVector tlate = transform.NetTranslation(); 

           data[0] = rot.xx();
           data[1] = rot.xy();
           data[2] = rot.xz();

           data[4] = rot.yx();
           data[5] = rot.yy();
           data[6] = rot.yz();

           data[8] = rot.zx();
           data[9] = rot.zy();
           data[10] = rot.zz();

           data[3] = tlate.x();
           data[7] = tlate.y();
           data[11] = tlate.z();

           data[12] = 0. ;
           data[13] = 0. ;
           data[14] = 0. ;
           data[15] = 1. ;
     }


   private:
       std::size_t   m_itemcapacity ; 
       std::size_t   m_itemsize ; 
       std::size_t   m_keysize ; 
       std::size_t   m_itemcount ; 
       double*       m_data ; 
       std::size_t*  m_key ; 


};


string transform_rep( G4AffineTransform& transform )
{
   G4RotationMatrix rotation = transform.NetRotation();
   G4ThreeVector rowX = rotation.rowX();
   G4ThreeVector rowY = rotation.rowY();
   G4ThreeVector rowZ = rotation.rowZ();
   G4ThreeVector tran = transform.NetTranslation(); 
   
   stringstream ss; 
   ss << tran << " " << rowX << rowY << rowZ  ;
   return ss.str();
}





int main()
{
  
   /* 
   G4AffineTransform t1(G4ThreeVector(1,2,3)) ; 
   G4AffineTransform t2(G4ThreeVector(10,20,30)) ; 

   TransformCache* tc = new TransformCache(2) ; 
   tc->Add( 1, t1 ); 
   tc->Add( 2, t2 ); 

   tc->Archive();
   */

   TransformCache* tc = TransformCache::Load();

   for( std::size_t n=0 ; n < tc->GetSize() ; ++n )
   {
       std::size_t key  = tc->GetKey(n);
       G4AffineTransform* find = tc->FindTransform(key);
       G4AffineTransform* transform = tc->GetTransform(n);
       if( transform ){
           cout << "transform " << n << " " << key << " "  << transform_rep(*transform) << endl ; 
       }
       if( find ){
           cout << "find      " << n << " " << key << " "  << transform_rep(*find) << endl ; 
       }
   }


   /*

   cnpy::NpyArray arr = cnpy::npy_load(name);
   const unsigned int ndims = arr.shape.size() ;

   unsigned int nels = 1;
   for(int i = 0;i < ndims;i++) nels *= arr.shape[i];

   printf("arr.shape.size() ndims %u \n", ndims );
   printf("arr.word_size %u \n", arr.word_size );


   for(int dim=0 ; dim < arr.shape.size() ; ++dim ){
       printf("arr.shape[%d] =  %u \n", dim, arr.shape[dim]);
   }

   printf("nels %d \n", nels );

   double* load = reinterpret_cast<double*>(arr.data);
   */



}
