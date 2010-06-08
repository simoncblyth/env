#ifndef CAPTURE_H
#define CAPTURE_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

using namespace std;

class Capture {
   private :
      stringstream m_ss   ;
      streambuf*   m_backup ;
   public :
      Capture(){
         m_backup = cout.rdbuf();    
         cout.rdbuf( m_ss.rdbuf() );
      }
      ~Capture(){
         cout.rdbuf( m_backup );
      }
      string Gotcha(){
         return m_ss.str(); 
      }
};


#endif
