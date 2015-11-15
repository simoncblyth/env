#include "icosahedron.h"
#include "NPY.hpp"

int main(int argc, char** argv)
{
     int level = 2 ; 
     int n = icosahedron_ntris(level);
     float* tris = icosahedron_tris(level);

     NPY<float>* buf = NPY<float>::make( n, 3, 3);
     buf->setData(tris);
     buf->save("/tmp/icosahedron.npy");

}
