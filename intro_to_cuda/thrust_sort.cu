
// nvcc thrust_sort.cu -run && rm a.out 
// nvcc -gencode arch=compute_37,code=sm_37 thrust_sort.cu -run && rm a.out 

#include <iostream>
#include <iomanip>

#include <thrust/host_vector.h> 
#include <thrust/device_vector.h> 
#include <thrust/sort.h>
#include <cstdlib>
       

void dump(thrust::host_vector<int>& vec,  unsigned i)
{
    std::cout << std::setw(15) << i 
              << " : " 
              << std::setw(15) << vec[i]
              << std::endl
              ;
}


void dump(thrust::host_vector<int>& vec,  unsigned num, const char* msg)
{
    unsigned size = vec.size();
    unsigned ndump = std::min(num, size) ;
    std::cout << msg << std::endl ; 

    for(unsigned i=0 ; i < ndump ; i++ ) dump(vec, i) ;
    for(unsigned i=vec.size() - ndump ; i < vec.size() ; i++ ) dump(vec, i) ;
    
}


int main(void)
{
    // generate 32M random numbers on the host
    unsigned size = 32 << 20 ;
    //unsigned size = 32  ;

    thrust::host_vector<int> h_vec(size); 
    thrust::generate(h_vec.begin(), h_vec.end(), rand);
    dump(h_vec, 32u, "generated");


    // transfer data to the device
    thrust::device_vector<int> d_vec = h_vec; // sort data on the device
    thrust::sort(d_vec.begin(), d_vec.end()); // transfer data back to host
    thrust::copy(d_vec.begin(), d_vec.end(), h_vec.begin()); 

    dump(h_vec, 32u, "sorted");


    return 0;
}


