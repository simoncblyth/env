// clang NPTest.cc -lc++ -o /tmp/NPTest

#include "NP.hh"

int main(int argc, char** argv)
{
    const char* path = argc > 1 ? argv[1] : "a.npy"  ; 
    assert( path ) ;

    NP<double> buf ; 
    buf.load(path); 

    buf.dump(0,1);
    buf.dump(999999,1000000);

    return 0 ; 
}
