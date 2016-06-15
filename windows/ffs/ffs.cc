// cmak-;cmak-cc ffs

#include <iostream>
#include <vector>

int ffs(int i) 
{ 
    // https://msdn.microsoft.com/en-us/library/wfd9z0bb.aspx
    unsigned long mask = i ; 
    unsigned long index ;
    unsigned char masknonzero = _BitScanForward( &index, mask );
    return masknonzero ? index + 1 : 0 ; 
} 


int main(int argc, char** argv)
{
    std::vector<int> msks ; 

    msks.push_back(0) ;
    msks.push_back(0x1) ;
    msks.push_back(0x10) ;
    msks.push_back(0x100) ;
    msks.push_back(0x1000) ;
    msks.push_back(0x10000) ;


    for(std::vector<int>::iterator it=msks.begin() ; it != msks.end() ; it++)
    { 
        int msk = *it ;
        int idx = ffs(msk) ;

        std::cout << " msk " << std::hex << msk << std::dec
                  << " idx " << idx
                 << std::endl ;  
    }

    return 0 ; 
}
