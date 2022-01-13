// name=chk ; gcc $name.cc -lstdc++ -o /tmp/$name && /tmp/$name

#include <cstdio>
int main()
{
   int n = 10 ; 
   for(int i=int(n);i--;) printf("%d\n", i) ; 
   return 0 ; 
}

