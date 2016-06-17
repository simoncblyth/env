#include "NPY.hpp"

int main()
{
    NPY<float>* npy = NPY<float>::make(10,1,4) ; 
    npy->zero();
    npy->save("/tmp/NPYClient.npy");

    return 0 ; 
}
