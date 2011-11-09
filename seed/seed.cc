#include <iostream>
using std::cout ;
using std::endl ;

#include <unistd.h>
#include <bitset>  
#include <string>  
#include <vector>  
#include <iomanip>  

long getseed(  long hostID , long runID , long evtID );
void strip(std::string& str );

void test_seed();
void test_strip();

int main(int argc,char** argv)
{
   //test_seed();
   test_strip();
}


void test_seed()
{
    long hostID( gethostid() ); 
    long runID = 0 ;
    for( int j = 0 ; j <=10 ; ++j ){
       long evtID((long)j) ;
       long seed = getseed( hostID , runID , evtID );  
       cout << "run:" << runID << " evt:" << evtID << " host:" << hostID << " seed:" << seed << endl ; 
    }
}


void test_strip()
{
   typedef std::vector<std::string> stringvec_t  ;
   stringvec_t trials ;

   trials.push_back("");
   trials.push_back("a");
   trials.push_back(" b");
   trials.push_back("c ");
   trials.push_back(" d ");
   trials.push_back(" hmmm ");
   trials.push_back("     Try a loada leading whitespce and no trailing") ;
   trials.push_back("     Try a loada leading whitespce and some trailing  ");
   trials.push_back("     Try a loada leading whitespce and one trailing ");
   trials.push_back(" One leading whitespace and one trailing ") ;
   trials.push_back("No leading whitespace and no trailing") ;
	
   stringvec_t::iterator itr ;
   stringvec_t::iterator beg = trials.begin();
   stringvec_t::iterator end = trials.end();

   itr = beg;
   for (; itr != end; ++itr) cout << std::setw(30) << "before[" << *itr << "]" << endl ; 

   itr = beg;
   for (; itr != end; ++itr) strip(*itr) ;

   itr = beg;
   for (; itr != end; ++itr) cout << std::setw(30) << "after[" << *itr << "]" << endl ; 
}


		
void strip(std::string& str )
{
  int i,j,z ;
  z = str.size() - 1;
  i = 0 ;
  while ( i<z && str[i] == ' ' ) ++i;    // increment i from 0 until hit non-whitespace 
  j = z ;
  while ( j>i && str[j] == ' ' ) --j ;   // decrement j from size until hit non-whitespace
  if ( i > 0 ) str.erase( 0, i ) ;
  if ( j < z ) str.erase( j - i + 1 ) ;  

  cout  << " i" << std::setw(5) << i 
        << " j" << std::setw(5) << j 
        << " z" << std::setw(5) << z 
        << endl ; 

}




long getseed( long hostID , long runID , long evtID ){
     
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



