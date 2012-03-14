/*
    port contents boost | grep program_options

    /opt/local/share/doc/boost/libs/program_options/example/multiple_sources.cpp

*/
#include <map>
#include <string>
#include <fstream>

#include <boost/program_options.hpp>
namespace po = boost::program_options;

#include <iostream>
#include <streambuf>

using namespace std;

typedef vector<string> svec ;
typedef map<string, string> ssmap;

// A helper function to simplify the main part.
template<class T>
ostream& operator<<(ostream& os, const vector<T>& v)
{
    copy(v.begin(), v.end(), ostream_iterator<T>(cout, " ")); 
    return os;
}


int dbxml_config(int argc, char **argv, po::variables_map& vm )
{
    try {
	     string config_file;

	     // Declare a group of options that will be allowed only on command line
	     po::options_description generic("Generic options");
	     generic.add_options()
		    ("version", "print version string")
		    ("help,h",    "produce help message")
  		    ("key,k",  po::value<svec>(),   "keys of variables propagated into XQuery scripts ")
		    ("val,v",  po::value<svec>(),   "vals of variables propagated into XQuery scripts ")
		    ("config,c", po::value<string>(&config_file)->default_value("hfagc.ini"), "name of a file of a configuration.")
		    ;

	     po::options_description config("Config file options");
	     config.add_options()
			("dbxml.environment_dir",    po::value<string>(), "dbxml environment dir ")
			("dbxml.default_collection", po::value<string>(), "default collection  ")
			("dbxml.baseuri",            po::value<string>(), "base uri  ")
			("container.path.path",      po::value<svec>(),   "container paths ")
			("container.tag.tag",        po::value<svec>(),   "container tags ")
			("namespace.name.name",      po::value<svec>(),   "namespace names ")
			("namespace.uri.uri",        po::value<svec>(),   "namespace uris ")
			;

	     po::options_description posit("Options that also work as positional arguments");
	     posit.add_options()
		    ("inputfile", po::value< vector<string> >(), "input file")
		    ;

	     po::options_description cmdline_options;
	     cmdline_options.add(generic).add(config).add(posit);

	     po::options_description config_file_options;
	     config_file_options.add(config);

	     po::options_description visible("Allowed options");
	     visible.add(generic).add(config).add(posit);
		
	     po::positional_options_description p;
	     p.add("inputfile", -1);
		
	     store(po::command_line_parser(argc, argv).options(cmdline_options).positional(p).run(), vm);
	     notify(vm);
		
	     ifstream ifs(config_file.c_str());
	     if (!ifs)
	     {
		   cout << "can not open config file: " << config_file << "\n";
		   return 0;
	     }
	     else
	     {
		   bool allow_unregistered = true ;
		   store(parse_config_file(ifs, config_file_options, allow_unregistered), vm);
		   notify(vm);
	     }
	    
	     if (vm.count("help")) {
		 cout << visible << "\n";
		 return 0;
	     }

	     if (vm.count("version")) {
		 cout << "Multiple sources example, version 1.0\n";
		 return 0;
	     }




   }
   catch(exception& e)
   {
        cout << e.what() << "\n";
        return 1;
   }    
   return 0;
}




/*
Config files such as:: 

	[container.path]
	path = /tmp/hfagc/hfagc.dbxml
	path = /tmp/hfagc/hfagc_system.dbxml

	[container.tag]
	tag = hfc
	tag = sys

Are converted into map with keys ("hfc","sys") and values ("/tmp/hfagc/hfagc.dbxml",..)

*/
void kv_multi_pluck( po::variables_map& vm , ssmap& ss , string kname , string vname )
{
      if (vm.count(kname) && vm.count(vname))
      {
           svec keys = vm[kname].as<svec>();
           svec vals  = vm[vname].as<svec>();
           size_t nkeys = keys.size() ; 
           if( nkeys == vals.size() ){
	       for(size_t i = 0 ; i < nkeys ; ++i )
               {
	            //cout << kname << " " << keys[i] << " " << vname << " " << vals[i] <<  endl ;
		    ss[keys[i]] = vals[i] ;
               }
           } else {
	       cout << "ERROR : numbers of " << kname << " and  " << vname << " must match " << endl ;
           } 		   
      }
}


/*
For single dotted keys such as dbxml.name from config such as::  

	  [dbxml]
	  name = a
	  other = b

Fill in the map with keys ("dbxml.name", "dbxml.other") and values ("a","b")

*/
void kv_pluck( po::variables_map& vm , ssmap& ss )
{
      po::variables_map::const_iterator it ;
      for( it = vm.begin() ; it != vm.end() ; ++it ){
         string key(it->first) ; 
         size_t ff = key.find(".");
         size_t rf = key.rfind(".");
         if( ff != string::npos && rf != string::npos && ff == rf ){    
	     ss[key] = (it->second).as<string>();		 
         }
      } 
}


void kv_dump( ssmap& ss )
{
   ssmap::const_iterator it ;
   for( it = ss.begin() ; it != ss.end() ; ++it ) cout << it->first << " : " << it->second << endl ;   
}


int main_dbxml_config(int argc, char **argv)
{
     po::variables_map vm;
     dbxml_config( argc , argv , vm );

     ssmap dbxml, containers, namespaces ;

     kv_pluck( vm , dbxml );
     kv_multi_pluck( vm , containers, "container.tag.tag", "container.path.path" );
     kv_multi_pluck( vm , namespaces, "namespace.name.name", "namespace.uri.uri" );

     kv_dump( dbxml );
     kv_dump( containers );
     kv_dump( namespaces );


     po::variables_map::const_iterator it;
     for( it = vm.begin() ; it != vm.end() ; ++it ) cout << it->first  << endl ;   



     /*
     ifstream t(xqpath.c_str()); 

     // ignore 1st line when 1st char is '#' allowing shebang running  
     char c = t.peek();
     if(c == '#') t.ignore( numeric_limits<streamsize>::max(), '\n' );  

     string q((istreambuf_iterator<char>(t)), istreambuf_iterator<char>());

     cout << q << endl ;
     */



}






