/*
   Command line tool to facilitate XQuerying dbxml container
   without having to escape the query.

  TODO:
     logging/verbosity control
     handle no inputfile 
     allow reading "inputfile" from stdin
     implicit DBEnv  ?
     comment (not scrub) first line
     writing output xml to file, when output path provided in options 
     
*/

#include <vector>
#include <string>
#include <map>
#include <fstream>
#include <iostream>
#include <streambuf>

#include "dbxml/DbXml.hpp"

#include "config.hh"
#include "extresolve.hh"
	
using namespace std;
using namespace DbXml;

typedef vector<string> svec ;
typedef map<string,string> ssmap ;
typedef map<string,ssmap> sssmap ;

int main(int argc, char **argv)
{
     sssmap cfg ;
     qxml_config( argc, argv, cfg );
       
     string xqpath( cfg["cli"]["inputfile"] );
     ifstream t(xqpath.c_str()); 
     char c = t.peek();
     if(c == '#') t.ignore( numeric_limits<streamsize>::max(), '\n' );  // ignore 1st line when 1st char is '#' allowing shebang running  
     string q((istreambuf_iterator<char>(t)), istreambuf_iterator<char>());

     //cout << q << endl ;

     DB_ENV* env = NULL;
     int dberr = db_env_create(&env, 0);
     if (dberr) {
	  cout << "Unable to create environment: " << db_strerror(dberr) << endl;
          if (env) env->close(env, 0);
          return EXIT_FAILURE;
     }

     u_int32_t env_flags = DB_CREATE | DB_INIT_MPOOL  ;
     const char *envHome = cfg["dbxml"]["dbxml.environment_dir"].c_str() ;  
     env->open(env, envHome, env_flags, 0);

     try {
        XmlManager mgr(env, DBXML_ALLOW_EXTERNAL_ACCESS)  ;
	MyResolver resolver;
	resolver.setXqmPath( cfg["dbxml"]["dbxml.xqmpath"] );
	mgr.registerResolver(resolver); 

        XmlContainer* cont = NULL ;
        ssmap::const_iterator it ;
        for( it = cfg["containers"].begin() ; it != cfg["containers"].end() ; ++it ){
	    cout << "containers:    " << it->first << " : " << it->second << endl ;   
            cont = new XmlContainer(mgr.openContainer( it->second ));   // hmm lodged in manager ?
            cont->addAlias(it->first) ;
        }

	XmlQueryContext qc = mgr.createQueryContext();        

	qc.setNamespace("my", resolver.getUri());
        qc.setDefaultCollection( cfg["dbxml"]["dbxml.default_collection"] );
        qc.setBaseURI( cfg["dbxml"]["dbxml.baseuri"]  );

        for( it = cfg["namespaces"].begin() ; it != cfg["namespaces"].end() ; ++it ){
	    cout << "namespaces:    " << it->first << " : " << it->second << endl ;   
	    qc.setNamespace( it->first, it->second);
        }
        for( it = cfg["variables"].begin() ; it != cfg["variables"].end() ; ++it ){
	    cout << "variables:    " << it->first << " : " << it->second << endl ;   
            qc.setVariableValue( it->first , it->second );
        }

        XmlResults res = mgr.query( q , qc);
        XmlValue value;
        while (res.next(value)) cout << "Value: " << value.asString() << endl;

    } catch (XmlException &e) {
         cout << "Exception: " << e.what() << std::endl;
    }
    return 0;
}
