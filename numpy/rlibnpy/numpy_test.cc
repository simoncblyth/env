//// clang numpy_test.cc -lstdc++ && ./a.out &&  ./load.py *_npy  


#include "numpy.hpp"
#include <iostream>
#include <cassert>

using namespace std ;

void onedim_a_npy()
{
   int aint[] = {1, 2, 3, 4, 5, 6};
   aoba::SaveArrayAsNumpy<int>(__func__, 6, &aint[0] );
}

void onedim_b_npy()
{
   cout << __func__ << endl ; 

   int aint[] = {1, 2, 3, 4, 5, 6};
   const int shape[] = {6} ;    
   bool fortran_order = true ; 
 

   aoba::SaveArrayAsNumpy<int>(__func__, true, 1, shape, &aint[0] );


   size_t nbytes = aoba::BufferSize<int>(1, shape, fortran_order  );
   char* buffer = new char[nbytes];
   size_t buflen = aoba::BufferSaveArrayAsNumpy<int>( buffer, true, 1, shape, &aint[0] ); 
   assert( buflen == nbytes );


   vector<int> oshape ; 
   vector<int> odata ; 
   aoba::BufferLoadArrayFromNumpy<int>( buffer, buflen, oshape, odata );  

   for( vector<int>::iterator it=oshape.begin() ; it != oshape.end() ; it++ ) cout << *it << " " ; 
   cout << endl ; 
   for( vector<int>::iterator it=odata.begin() ; it != odata.end() ; it++ ) cout << *it << " " ;
   cout << endl ; 
}

void twodim_a_npy()
{
   cout << __func__ << endl ; 

   int points[][3] = {{ 1, 2, 3 },
                      { 4, 5, 6 },
                      { 7, 8, 9},
                      {10,11,12}};

   //aoba::SaveArrayAsNumpy<int>(__func__, 4, 3, &points[0][0] );

   bool fortran_order = false ; 
   const int shape[] = {4, 3} ;    
   size_t nbytes = aoba::BufferSize<int>(2, shape, fortran_order  );

   char* buffer = new char[nbytes];
   size_t buflen = aoba::BufferSaveArrayAsNumpy<int>( buffer, fortran_order, 2, shape, &points[0][0] ); 
   assert( buflen == nbytes );


   vector<int> oshape ; 
   vector<int> odata ; 
   aoba::BufferLoadArrayFromNumpy<int>( buffer, buflen, oshape, odata );  

   for( vector<int>::iterator it=oshape.begin() ; it != oshape.end() ; it++ ) cout << *it << " " ; 
   cout << endl ; 
   for( vector<int>::iterator it=odata.begin() ; it != odata.end() ; it++ ) cout << *it << " " ;
   cout << endl ; 

}

void twodim_b_npy()
{
   int points[][3] = {{ 1, 2, 3 },
                      { 4, 5, 6 },
                      { 7, 8, 9},
                      {10,11,12}};

   const int shape[] = {4, 3} ;    
   aoba::SaveArrayAsNumpy<int>(__func__, true, 2, shape, &points[0][0] );
}


void flexi_npy()
{
   int points[][4][3] = {
                      {
                        { 1, 2, 3 },
                        { 4, 5, 6 },
                        { 7, 8, 9 },
                        {10,11,12 }
                      },
                      {
                        { 10, 20, 30 },
                        { 40, 50, 60 },
                        { 70, 80, 90 },
                        {100,110, 120 }
                      },
                      {
                        { 1, 2, 3 },
                        { 4, 5, 6 },
                        { 7, 8, 9 },
                        {10,11,12 }
                      }
                     };

   aoba::SaveArrayAsNumpy<int>(__func__, 3, "4,3",  &points[0][0][0] );
}




int main()
{
   //onedim_a_npy();
   //onedim_b_npy();
   //twodim_a_npy();
   //twodim_b_npy();

   flexi_npy();

   return 0 ;
}
