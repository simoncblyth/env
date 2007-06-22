#include <iostream>
using std::cout ;
using std::endl ;

#include <unistd.h>
#include <bitset>  

int main(int argc,char** argv)
{
    // extracting pieces from   dyw_2_9:dywPrimaryGeneratorAction.cc
    
    long hostID( gethostid() );
	long runID( 0 );
    long eventID( 1 );
    
    std::bitset<32> run ( runID );
    std::bitset<32> site ( hostID );
    std::bitset<32> evt ( evtID );
    std::bitset<32> a ( (long) 0);
    
    // loop over bits in run in reverse order and set bits in a
    for ( int i = 32-1; i >= 0 ; i-- )
    {
        if ( run.test(i) ) a.set(31-i) ;
    }
    
    // create seed = (a OR evt) XOR site
    std::bitset<32> seed = (a|evt)^site ; 
    
    // set highest bit to zero to avoid negative seed
    if ( seed.test(31) ) seed.reset(31) ; 
    
        
    long myseed( seed.to_ulong() );  
  
    
    cout << "run/evt/host/seed" << runID << evtID << hostID << myseed << endl ;  
    
    

}
								  

