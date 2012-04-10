/*
    port contents boost | grep program_options

    /opt/local/share/doc/boost/libs/program_options/example/multiple_sources.cpp

*/
#include <fstream>
#include <iostream>
#include <streambuf>

#include "potools.hh"

#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/fstream.hpp>
	
//using namespace boost::filesystem; 

template<class T>
ostream& operator<<(ostream& os, const vector<T>& v)
{
    copy(v.begin(), v.end(), ostream_iterator<T>(cout, " ")); 
    return os;
}

int prepare_dir( const string& target )
{
   try
   {	   
      if( !boost::filesystem::exists( target ) )
      {
	  bool ok = boost::filesystem::create_directory( target );
          if(!ok) cerr << "preparedir failed to create " << target << endl ;	  
      }
   }
   catch (const boost::filesystem::filesystem_error& ex)
   {
	cerr << ex.what() << '\n';
   }
   return 0 ;
}

int qxml_popvm(int argc, char **argv, po::variables_map& vm )
{
    try {
	     string config_file;

             po::options_description envvar("Envvar options");
             envvar.add_options()
		("config,c", po::value<string>(&config_file)->default_value("hfagc.ini"), "name of a file of a configuration.")
		;

	     // Declare a group of options that will be allowed only on command line
	     po::options_description generic("Generic options");
	     generic.add_options()
		("version", "print version string")
		("help,h",  "produce help message")
  		("key,k",    po::value<svec>(),   "keys of variables propagated into XQuery scripts ")
		("val,v",    po::value<svec>(),   "vals of variables propagated into XQuery scripts ")
		;

	     po::options_description config("Config file options");
	     config.add_options()
		("dbxml.environment_dir",    po::value<string>(), "dbxml environment dir ")
		("dbxml.default_collection", po::value<string>(), "default collection  ")
		("dbxml.baseuri",            po::value<string>(), "base uri  ")
		("dbxml.xqmpath",            po::value<string>(), "comma delimited module directories ")
		("container.srcdir.srcdir",  po::value<svec>(),   "container source directories ")
		("container.path.path",      po::value<svec>(),   "container paths ")
		("container.tag.tag",        po::value<svec>(),   "container tags ")
		("namespace.name.name",      po::value<svec>(),   "namespace names ")
		("namespace.uri.uri",        po::value<svec>(),   "namespace uris ")
		;

	     po::options_description posit("Options that also work as positional arguments");
	     posit.add_options()
		("inputfile", po::value<string>(), "input file")
		;

	     po::options_description envvar_options;
	     envvar_options.add(envvar);

	     po::options_description cmdline_options;
	     cmdline_options.add(generic).add(config).add(posit);

	     po::options_description config_file_options;
	     config_file_options.add(config);

	     po::options_description visible("Allowed options");
	     visible.add(generic).add(config).add(posit);

	   
	     po::positional_options_description p;
	     p.add("inputfile", -1);
		
	     //po::parsed_options parsed = po::command_line_parser(argc, argv).options(cmdline_options).positional(p).allow_unregistered().run() ;
	     // allowing unregisted presents problems with positionals
	     
	     po::parsed_options parsed = po::command_line_parser(argc, argv).options(cmdline_options).positional(p).run() ;


	     // NB somewhat bizarre : if an envvar such as QXML_ENTITYPATH were defined this would
	     // yield an error :  
             po::parsed_options eparse = po::parse_environment(envvar, "QXML_"); 

	     store( parsed, vm);
	     store( eparse, vm);
	     notify(vm);

             //cout << "parse environment " << eparse.options.size() << " config_file" << config_file << endl ;

	     //svec unrec = po::collect_unrecognized(parsed.options, po::include_positional);
             //svec::const_iterator it ;
             //for( it = unrec.begin() ; it != unrec.end() ; ++it ) cout << "unrec " << *it<< endl ;   

	     ifstream ifs(config_file.c_str());
	     if (!ifs)
	     {
		   cout << "can not open config file: " << config_file << "\n";
		   return 0;
	     }
	     else
	     {
	           bool allow_unregistered = false ;
		   store(parse_config_file(ifs, config_file_options, allow_unregistered), vm);
		   notify(vm);
	     }
	    
	     if (vm.count("help")) {
		 cout << visible << "\n";
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




int qxml_config(int argc, char **argv , sssmap& m )
{
     po::variables_map vm;
     qxml_popvm( argc , argv , vm );

     kv_pluck(       vm , m["dbxml"] , 1 );
     kv_pluck(       vm , m["cli"]   , 0 );

     kv_zip( vm , m["containers"], "container.tag.tag", "container.path.path" );
     kv_zip( vm , m["namespaces"], "namespace.name.name", "namespace.uri.uri" );
     kv_zip( vm , m["variables"] , "key",                 "val" );

     //vm_dump( vm );
     //cfg_dump(m);
}





