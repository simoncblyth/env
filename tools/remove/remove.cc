// clang remove.cc -o /tmp/remove -lc++ 

#include <iostream>
#include <cstdio>

int main(int argc, char** argv)
{
    // hmm this will remove directories too
    // and it seems non-trivial to cross platform distinguish 
    // dirs from files
    //
    // the right thing to do is to just use boost::filesystem 
    // BUT that is probably not-permissable in G4DAE code 

    if(argc != 2) 
    {
       std::cerr << "Expecting single filesystem path argument of the file to remove " << std::endl ;  
       return 0 ;  
    }
    const char* path = argv[1] ; 
  
    std::cout << "removing " << path << std::endl ; 

    int rc = std::remove(path) ;

    if(rc!=0) std::cerr << "error removing path " << path << " rc " << rc  << std::endl ; 
    
    return rc ;
}
