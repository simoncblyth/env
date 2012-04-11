/*
   Command line tool to facilitate XQuerying dbxml container
   without having to escape the query.

  TODO:
     logging/verbosity control
     allow reading "inputfile" from stdin
     implicit DBEnv  ?
     comment (not scrub) first line
     writing output xml to file, when output path provided in options 
     
*/

#include <boost/chrono.hpp>

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

/*
*/

int main(int argc, char **argv)
{
     boost::chrono::system_clock::time_point t_start, t_prequery, t_postquery, t_end ;
     t_start = boost::chrono::system_clock::now();

     sssmap cfg ;
     qxml_config( argc, argv, cfg );
       
     string loglevel( cfg["cli"]["level"] );  // TODO: find C++ logging approach 
     string xqpath( cfg["cli"]["inputfile"] );
     ifstream t(xqpath.c_str()); 
     char c = t.peek();
     if(c == '#') t.ignore( numeric_limits<streamsize>::max(), '\n' );  // ignore 1st line when 1st char is '#' allowing shebang running  
     string q((istreambuf_iterator<char>(t)), istreambuf_iterator<char>());

     //cout << q << endl ;

     DB_ENV* env = NULL;
     int dberr = db_env_create(&env, 0);
     if (dberr) {
	  cerr << "Unable to create environment: " << db_strerror(dberr) << endl;
          if (env) env->close(env, 0);
          return EXIT_FAILURE;
     }

     u_int32_t env_flags = DB_CREATE | DB_INIT_MPOOL  ;
     string envdir = cfg["dbxml"]["dbxml.environment_dir"] ;  
     prepare_dir( envdir );
     const char *envHome = envdir.c_str() ;  
     env->open(env, envHome, env_flags, 0);

     try {
        XmlManager mgr(env, DBXML_ALLOW_EXTERNAL_ACCESS)  ;
	MyResolver resolver;
	resolver.setXqmPath( cfg["dbxml"]["dbxml.xqmpath"] );
	mgr.registerResolver(resolver); 


        XmlContainer* cont = NULL ;
        ssmap::const_iterator it ;
        for( it = cfg["containers"].begin() ; it != cfg["containers"].end() ; ++it ){
	    string tag = it->first ;     
	    string name = it->second ;     
            int chk = mgr.existsContainer(name);
	    clog << "containers:    " << tag << " : " << name << " ? " << chk << endl ;   

	    XmlContainer* cont ;
            if(chk == 0){
                cont = new XmlContainer(mgr.createContainer(name));
	    } else {	    
		cont = new XmlContainer(mgr.openContainer(name));   
	    }               	
	    cont->addAlias(tag);
            //	    
	    // need to "leak" containers to avoid evaporation 
	    //       Exception: Error: Cannot resolve container: hfc.  
	    //       Container not open and auto-open is not enabled.  Container may not exist.
            //

            if( tag == "tmp" ){
		 //   
		 // give special meaning to container tag "tmp" , 
		 // some external functions need such a container for scratch space     
		 //
		 //clog << "setting tmpContainer and Name " << endl ;
	         resolver.setTmpContainer(name);             // dbxml uri of the scratch container
	         resolver.setTmpName("tmpfragment.xml");     // default name of docs created there 
	    }

        }

    
	resolver.readGlyphs(mgr);
	//resolver.dumpGlyphs();

	XmlQueryContext qc = mgr.createQueryContext();        

	qc.setNamespace("my", resolver.getUri());
        qc.setDefaultCollection( cfg["dbxml"]["dbxml.default_collection"] );
        qc.setBaseURI( cfg["dbxml"]["dbxml.baseuri"]  );

        for( it = cfg["namespaces"].begin() ; it != cfg["namespaces"].end() ; ++it ){
	    clog << "namespaces:    " << it->first << " : " << it->second << endl ;   
	    qc.setNamespace( it->first, it->second);
        }
        for( it = cfg["variables"].begin() ; it != cfg["variables"].end() ; ++it ){
	    clog << "variables:    " << it->first << " : " << it->second << endl ;   
            qc.setVariableValue( it->first , it->second );
        }

        t_prequery = boost::chrono::system_clock::now();
        XmlResults res = mgr.query( q , qc);
        t_postquery = boost::chrono::system_clock::now();
        XmlValue value;
	// NB the only **cout**, to make it possible to output valid XML
        while (res.next(value)) cout << value.asString() << endl;

    } catch (XmlException &e) {
         cerr << "Exception: " << e.what() << std::endl;
    }
    t_end = boost::chrono::system_clock::now();


    boost::chrono::duration<double> d_input = t_prequery - t_start ;
    boost::chrono::duration<double> d_query = t_postquery - t_prequery ;
    boost::chrono::duration<double> d_output = t_end - t_postquery ;
    boost::chrono::duration<double> d_total  = t_end - t_start ;

    clog << "total  " << d_total.count()  << " s\n";
    clog << "input  " << d_input.count()  << " s\n";
    clog << "query  " << d_query.count()  << " s\n";
    clog << "output " << d_output.count() << " s\n";
    return 0;
}
