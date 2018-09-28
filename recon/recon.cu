// nvcc -I$HOME/np recon.cu && ./a.out && rm a.out 

#include <thrust/device_vector.h>
#include "NP.hh"

template<typename T>
struct SumFunc 
{
    __device__ T operator()(T val) const 
    {
       return 2.*val ;  
    }
};

template<typename T>
struct Recon
{
    NP<T>* t ; 
    NP<T>* sph ; 
    thrust::device_vector<T> d_t ; 
    thrust::device_vector<T> d_sph ; 

    Recon(const char* dir); 

    //T nll();  
    T sum();  
};

template<typename T>
Recon<T>::Recon(const char* dir)  
    :
    t(NP<T>::Load(dir, "t.npy")), 
    sph(NP<T>::Load(dir, "sph.npy")),
    d_t(t->data.begin(), t->data.end()),
    d_sph(sph->data.begin(), sph->data.end())
{
    t->dump(0,10); 
    sph->dump(0,10); 
} 

/*
template<typename T>
T Recon<T>::nll()
{
    NllFunc<T> nllfunc ; 
    return thrust::transform_reduce( 
                 thrust::make_zip_iterator(thrust::make_tuple(d_t.begin(), d_sph.begin())),
                 thrust::make_zip_iterator(thrust::make_tuple(d_t.end(),   d_sph.end())),
                 nllfunc, 
                 T(0),   
                 thrust::plus<T>());
} 
*/

template<typename T>
T Recon<T>::sum()
{
    SumFunc<T> sumfunc ; 
    return thrust::transform_reduce( d_t.begin(), d_t.end(), sumfunc, T(0), thrust::plus<T>() ) ;
}
 
int main(int argc, char** argv)
{
    const char* dir = argc > 1 ? argv[1] : "/tmp/recon" ; 

    Recon<double> rec(dir) ;  

    std::cout << "rec.sum " << rec.sum() << std::endl ;     

    return 0 ; 
}

