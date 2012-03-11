/*
 
   Command line tool to facilitate XQuerying dbxml container
   without having to escape the query.


*/
#include <string>
#include <fstream>
#include "dbxml/DbXml.hpp"

#include <boost/program_options.hpp>
namespace po = boost::program_options;

#include <iostream>
#include <streambuf>

#include "extfun.hh"
	
using namespace std;
using namespace DbXml;

typedef vector<string> svec ;


int main(int argc, char **argv)
{
      string xqpath("");
      string dbxmldir("/tmp/hfagc") ;
      //string baseuri("dbxml:/") ;
      string baseuri("") ;
      svec keys ;
      svec vals ;

      po::options_description desc("Allowed options");
      desc.add_options()
		("help,h",        "produce help message")
		("xqpath,q",      po::value(&xqpath), "path for input xquery, positional argument also works ")
		("baseuri,b",     po::value(&baseuri), "baseuri ")
		("key,k",        po::value<svec>(&keys), "keys ")
		("val,v",        po::value<svec>(&vals), "vals ")
		("dbxmldir,d",    po::value(&dbxmldir), "path of directory containing hfagc.dbxml and hfagc_system.dbxml containers")
		;

      po::positional_options_description p;
      p.add("xqpath", -1);

      po::variables_map vm;
      try{
          po::parsed_options parsed = po::command_line_parser(argc, argv).options(desc).positional(p).run();
          po::store(parsed, vm);
          po::notify(vm);
      } catch (exception &e) {
          cout << "Exception: " << e.what() << endl;
          return 1;
      }


      if (vm.count("help") || xqpath == "") {
           cout << desc << "\n";
	   return 1;
      }


      size_t nkeys = keys.size() ;
      size_t nvals = vals.size() ;
      if(nkeys != nvals){
	    cout << "ERROR : number of keys must match the number of vals " << endl ;
	    return 2 ;  
      } 

      for(size_t i = 0 ; i < nkeys ; ++i )
      {
	   cout << "key " << keys[i] << " " << vals[i] <<  endl ;
      }

     ifstream t(xqpath.c_str()); 

     // ignore 1st line when 1st char is '#' allowing shebang running  
     char c = t.peek();
     if(c == '#') t.ignore( numeric_limits<streamsize>::max(), '\n' );  

     string q((istreambuf_iterator<char>(t)), istreambuf_iterator<char>());

     cout << q << endl ;
 

     DB_ENV* env = NULL;
     int dberr = db_env_create(&env, 0);
     if (dberr) {
	  cout << "Unable to create environment: " << db_strerror(dberr) << endl;
          if (env) env->close(env, 0);
          return EXIT_FAILURE;
     }

     u_int32_t env_flags = DB_CREATE | DB_INIT_MPOOL  ;
     char *envHome = "/tmp/dbxml";  
     env->open(env, envHome, env_flags, 0);

     try {
        XmlManager mgr(env, DBXML_ALLOW_EXTERNAL_ACCESS)  ;

	// Create an function resolver
	MyFunResolver resolver;

	// Register the function resolver to XmlManager
	mgr.registerResolver(resolver); 

        
        XmlContainer hfagc = mgr.openContainer( dbxmldir + "/hfagc.dbxml");
        hfagc.addAlias("hfc") ;

        XmlContainer sys   = mgr.openContainer( dbxmldir + "/hfagc_system.dbxml");
        sys.addAlias("sys") ;

        XmlQueryContext qc = mgr.createQueryContext();        

	qc.setNamespace("rez","http://hfag.phys.ntu.edu.tw/hfagc/rez");

	// Set the prefix URI
	qc.setNamespace("my", resolver.getUri());

        qc.setDefaultCollection("dbxml:///" + dbxmldir + "/hfagc.dbxml");
        qc.setBaseURI( baseuri );

	// populate context with key value pairs   
   	for(size_t i = 0 ; i < nkeys ; ++i )
        {
	   cout << "  $" << keys[i] << " := \"" << vals[i] << "\"" << endl ;
           qc.setVariableValue( keys[i] , vals[i] );
        }

        XmlResults res = mgr.query( q , qc);

        // Print out the result of the query
        XmlValue value;
        while (res.next(value)) cout << "Value: " << value.asString() << endl;

    } catch (XmlException &e) {
         cout << "Exception: " << e.what() << std::endl;
    }

    return 0;
}
