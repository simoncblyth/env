#include "NPY.hpp"

int main()
{

    NPYBase::setGlobalVerbose();

    const char* path = "/tmp/NPYClient.npy" ; 


    unsigned int N = 10 ; 

    NPY<float>* a = NPY<float>::make(N,1,4) ; 
    a->zero();

    for(unsigned int i=0; i < N ; i++) a->setQuad(i,0,0, i*10 + 0,i*10+1, i*10+2, i*10+3) ;
    a->save(path);


    NPY<float>* b = NPY<float>::load(path);
    b->dump();


    return 0 ; 
}
