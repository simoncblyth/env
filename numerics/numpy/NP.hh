#include <iostream>
#include <string>
#include <vector>
#include <cassert>
#include <fstream>

template<typename T>
struct NP
{
    NP() : ni(0), nj(0), nk(0) {} ; 

    void load(const char* path);   
    void dump(unsigned i0, unsigned i1) const ;   

    std::vector<T> data ; 
    size_t ni ; 
    size_t nj ; 
    size_t nk ; 
};

template<typename T>
void NP<T>::load(const char* path)
{
    std::cout << "NP::load " << path << std::endl ; 

    std::ifstream stream(path, std::ios::in|std::ios::binary);

    std::string header(128, ' '); 
    stream.read(&header[0], 128 ); 

    std::cout << header << std::endl ; 

    // a real implementation would parse the header to get the shape 
    ni = 1000000 ; 
    nj = 4 ; 
    nk = 4 ; 

    size_t total_items = ni*nj*nk ;  
    size_t total_bytes = total_items*sizeof(T) ; 

    data.resize(total_items);

    stream.read(reinterpret_cast<char*>(&data[0]), total_bytes );
}


template<typename T>
void NP<T>::dump(unsigned i0, unsigned i1) const 
{
    for(unsigned i=i0 ; i < i1 ; i++){
        std::cout << "[" << i  << "]" << std::endl ;  
        for(unsigned j=0 ; j < nj ; j++){
            for(unsigned k=0 ; k < nk ; k++)
            {
                size_t index = i*nj*nk + j*nk + k ; 
                std::cout << " " << std::fixed << data[index] ;      
            }
            std::cout << std::endl ; 
        }
        std::cout << std::endl ; 
    }
}

