// clang tie_tuple.cc  -std=c++11 -lc++ -o /tmp/tie_tuple && /tmp/tie_tuple 

#include <iostream>
#include <tuple>  

void dump(const int& x, const int& y)
{
   std::cout 
      << " x : " << x 
      << " y : " << y
      << std::endl ; 
      ; 
}

void test_swap()
{
    int x = 5, y = 7;
    dump(x,y);

    std::tie(x,y) = std::make_tuple(y,x);
    dump(x,y);
    // swap happens because tie uses references, but tuple copies 
}


int main(int argc,char** argv)
{
    test_swap() ;
    return 0 ; 
}


