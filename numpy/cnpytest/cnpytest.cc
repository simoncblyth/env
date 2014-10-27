#include "cnpy.h"

#include <cstddef>

#include "G4ThreeVector.hh"
#include "G4AffineTransform.hh"

#include <iostream>

using namespace std ; 



class TransformCache {
  public:
     TransformCache( std::size_t itemcapacity ) : 
             m_itemcapacity(itemcapacity), 
             m_itemsize(4*4), 
             m_keysize(1), 
             m_data(0), 
             m_key(0), 
             m_itemcount(0) 
     {
         m_data = new double[m_itemcapacity*m_itemsize] ;
         m_key  = new std::size_t[m_itemcapacity] ;
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
        std::size_t* key = reinterpret_cast<std::size_t*>(akey.data);

        for(std::size_t n=0 ; n < akey.shape[0] ; n++ ){
           cout << "akey " << n << " => " << key[n] << endl ;  
        }

        cnpy::NpyArray adata = cnpy::npy_load("data.npy"); 
        assert( adata.shape.size() == 3 );
        assert( adata.shape[1] == 4 && adata.shape[2] == 4 );
   
        double* data = reinterpret_cast<double*>(adata.data);

        for(std::size_t n=0 ; n < adata.shape[0]*4*4 ; n++ ){
           cout << "adata " << n << " => " << data[n] << endl ;  
        }

        return NULL ;  
    }


     void Add( std::size_t identifier, const G4AffineTransform&  transform )
     {
           assert(m_itemcount < m_itemcapacity );

           double* data = m_data + m_itemcount*m_itemsize ;   
           std::size_t* id = m_key + m_itemcount*m_keysize ; 

           m_itemcount++ ; 

           id[0] = identifier ; 
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

   TransformCache* load = TransformCache::Load();


 


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
