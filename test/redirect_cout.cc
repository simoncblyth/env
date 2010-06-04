/*
   http://www.cplusplus.com/reference/iostream/ios/rdbuf/

   redirecting cout's output

*/
#include <iostream>
#include <fstream>

#include <sstream>
#include <string>

using namespace std;

void cout_to_file()
{
  streambuf *psbuf, *backup;
  ofstream filestr;
  filestr.open ("test.txt");

  backup = cout.rdbuf();     // back up cout's streambuf

  psbuf = filestr.rdbuf();   // get file's streambuf
  cout.rdbuf(psbuf);         // assign streambuf to cout

  cout << "This is written to the file";
  
  cout.rdbuf(backup);        // restore cout's original streambuf

  filestr.close();
}


void cout_to_string()
{
  streambuf *psbuf, *backup;
  stringstream sstr ;
  backup = cout.rdbuf();     // back up cout's streambuf
  psbuf = sstr.rdbuf();      // get file's streambuf
  cout.rdbuf(psbuf);         // assign streambuf to cout

  cout << "This is written to the stringstream ";
  
  cout.rdbuf(backup);        // restore cout's original streambuf

  string gotcha = sstr.str();
  cout << "did we : " << gotcha << endl ;
}


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

void cout_to_string_oo_1()
{
  Capture* c = new Capture(); 
  cout << "This is written to the stringstream ";
  string gotcha = c->Gotcha();
  delete c ; 
  cout << "did we : " << gotcha << endl ;
}

void cout_to_string_oo_2()
{
  // let the scoping do our deletion 
  string gotcha ;
  {
      Capture c; 
      cout << "This is written to the stringstream ";
      gotcha = c.Gotcha();
  }
  cout << "did we : " << gotcha << endl ;
}
int main () {

  cout_to_file();
  cout_to_string();
  cout_to_string_oo_1();
  cout_to_string_oo_2();

  return 0;
}


