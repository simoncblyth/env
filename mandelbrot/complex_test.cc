// name=complex_test ; gcc $name.cc -I/usr/local/cuda/include -I/usr/local/opticks/include/SysRap -std=c++11 -lstdc++ -o /tmp/$name && /tmp/$name

#include <complex>
#include <iostream>
#include <iomanip>

void complex_test()
{
    typedef std::complex<double> Complex_t ; 

    Complex_t z0(1.,1.) ; 
    Complex_t z(z0) ; 

    for(unsigned i=0 ; i < 10 ; i++ )
    {
        std::cout 
             << " i " << std::setw(3) << i 
             << " z " << std::setw(20) << z 
             << " std::norm(z) " << std::setw(12) << std::norm(z) 
             << " std::abs(z) "  << std::setw(12) << std::abs(z) 
             << " std::pow(z0,i) "  << std::setw(12) << std::pow(z0, i) 
             << std::endl 
             ;  
        z = z*z ;     
    }
}

int main(int argc, char** argv)
{
    complex_test(); 
  
    return 0 ; 
}
