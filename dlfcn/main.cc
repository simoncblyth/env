// http://www.linuxjournal.com/article/3687
#include <dlfcn.h>
#include <iostream>
#include <string>

#include "shape.hh"

using namespace std ;


// our global factory for making shapes
map<string, maker_ptr*, less<string> > factory;

int main(int argc, char **argv)
{
   const char* lib  = argv[1] ;
   const char* name = argv[2] ;

   void* dlib = dlopen(lib, RTLD_NOW);
   if(dlib == NULL){
         cerr << "failed to open " << lib << endl ;
	 cerr << dlerror() << endl;
	 exit(-1);
   } else {
         clog << "opened " << lib << endl ;
   }
  
  /* 
   // casting a void* into a function of void 
   // that returns a pointer to shape* is ugly
   // and tricky to get correct
   void* mkr = dlsym(hndl, "maker");
   clog << "mkr " << mkr << endl ;
   shape *sh = ((shape*(*)())(mkr))();    
   //shape* sh = ((maker_t    )(mkr))();
   //shape* sh = reinterpret_cast<shape* ()>(mkr);   fails to compile
   //shape* sh = function_cast<maker_t>(mkr);
   */

   //let the registry do the dirty work
   shape* sh = factory[name]() ;
   sh->draw(); 


   dlclose( dlib );

}

