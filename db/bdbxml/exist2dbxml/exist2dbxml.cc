/*
 
  Creates Berkeley DB XML container and inserts all xml files
  (eg from an eXist backup) beneath a root directory into it, 
  with root relative names.


  bdbxml embeds XercesC so there could be conflict with the macports one

  g++ -I$BDBXML_HOME/include -I/opt/local/include -c migrate.cc 
  g++ -L$BDBXML_HOME/lib -ldbxml   -L/opt/local/lib -lboost_filesystem -lboost_system migrate.o -o migrate 

  http://www.boost.org/doc/libs/1_49_0/libs/filesystem/v3/doc/index.htm
  http://www.boost.org/doc/libs/1_49_0/libs/filesystem/v3/doc/tutorial.html
  http://www.boost.org/doc/libs/1_48_0/libs/filesystem/v3/doc/reference.html#path-decomposition

 */

#include <string>
#include <fstream>
#include "dbxml/DbXml.hpp"

#include <vector>
#include <iostream>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/fstream.hpp>

using namespace boost::filesystem; 
using namespace std ;
using namespace DbXml;

typedef vector<string> vec;    


// recursively collect paths  	
void listdir( vec& paths, const path& directory, bool recurse_into_subdirs = true )
{
   try
   {	   
      if( exists( directory ) )
      {
          directory_iterator end ;
          for( directory_iterator iter(directory) ; iter != end ; ++iter ){
	      if(is_directory(*iter)){
	          //cout << *iter << " (directory)\n" ;
  		  if( recurse_into_subdirs ) listdir( paths, *iter) ;
	       } else {	   
		  paths.push_back( iter->path().native() );      
	       }
           }	   
       }
   }
    catch (const filesystem_error& ex)
    {
	cout << ex.what() << '\n';
    }
}


void ingest( string root , string dbxml )
{
    try {
        XmlManager mgr;
        XmlContainer cont = mgr.createContainer(dbxml);
        XmlUpdateContext ctx = mgr.createUpdateContext(); 

        vec v;
        listdir( v, root ) ;
        for (vec::const_iterator it (v.begin()); it != v.end(); ++it)
        {
             string p = *it ;
             string n = p.substr(root.length()+1) ;       // use name relative to root 
             size_t found = n.find("__contents__.xml");   // skip eXist metadata files
	     if(found == string::npos ){
		   cout << "   " << p << " " << n << " " << '\n';
		   XmlInputStream* stm = mgr.createLocalFileInputStream(p);
                   cont.putDocument( n, stm , ctx , 0); 
             }
	}

    } catch (XmlException &e) {
        std::cout << "Exception: " << e.what() << std::endl;
    }
}



int main(int argc, char* argv[]) 
{

     // TODO: adopt boost_program_options here, see qxml for how	
    string source="/data/heprez/data/backup/part/localhost/2012/Mar06-1922" ;
    string target="/tmp/hfagc" ; 

    ingest( source + "/db/hfagc"        , target + "/hfagc.dbxml" );
    ingest( source + "/db/hfagc_system" , target + "/hfagc_system.dbxml" );

    return 0;
}


