#include <iostream>
using std::cout ;
using std::endl ;

#include <unistd.h>


int main(int argc,char** argv)
{
    
    long hostID = gethostid();
    
	cout << "seed/hostid checking ... " << hostID << endl ;							      
}
								  

