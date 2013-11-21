#ifndef _G4DAEUTIL_INCLUDED_
#define _G4DAEUTIL_INCLUDED_

#include <iostream>
#include <vector>
#include <string>
#include <streambuf>

class G4DAEUtil 
{
public:

   static void replaceAll(std::string& id, std::string const& from, std::string const& to);
   static int transformNCName( std::string& id, bool encode );
   static int decodeNCName( std::string& id );
   static int encodeNCName( std::string& id );
   static int testNCName( std::string& id );
   static int testNCNameDemo();
   static void WriteLines(std::string& fname, std::vector<std::string>& lines);

};



// http://stackoverflow.com/questions/5419356/redirect-stdout-stderr-to-a-string

struct cout_redirect {
    cout_redirect( std::streambuf * new_buffer ) 
        : old( std::cout.rdbuf( new_buffer ) )
    { }

    ~cout_redirect( ) {
        std::cout.rdbuf( old );
    }

private:
    std::streambuf * old;
};


struct cerr_redirect {
    cerr_redirect( std::streambuf * new_buffer ) 
        : old( std::cerr.rdbuf( new_buffer ) )
    { }

    ~cerr_redirect( ) {
        std::cerr.rdbuf( old );
    }

private:
    std::streambuf * old;
};




#endif

