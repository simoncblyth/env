#include <iostream>
using std::cout ;
using std::endl ;

#include <unistd.h>
#include <bitset>  

long seed(  long hostID , long runID , long evtID );

int main(int argc,char** argv)
{
    long hostID( gethostid() ); 
    long runID = 0 ;
    for( int j = 0 ; j <=10 ; ++j ){
       long evtID((long)j) ;
       long seed = seed( hostID , runID , evtID );  
       cout << "run:" << runID << " evt:" << evtID << " host:" << hostID << " seed:" << seed << endl ; 
    }
}
			

long seed( long hostID , long runID , long evtID ){
     
    // extracting pieces from   dyw_2_9:dywPrimaryGeneratorAction.cc
    
    std::bitset<32> site ( hostID );
    std::bitset<32> run ( runID );
    std::bitset<32> evt ( evtID );
    
     // loop over bits in run in reverse order and set bits in a
    std::bitset<32> a ( (long) 0);
    for ( int i = 32-1; i >= 0 ; i-- )
    {
        if ( run.test(i) ) a.set(31-i) ;
    }
    
    // create seed = (a OR evt) XOR site
    std::bitset<32> seed = (a|evt)^site ; 
    
    // set highest bit to zero to avoid negative seed
    if ( seed.test(31) ) seed.reset(31) ; 
    
    return seed.to_ulong()  ;
}



