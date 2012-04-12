/*
   Command line tool to facilitate XQuerying dbxml container
   without having to escape the query.

  TODO:
     logging/verbosity control
     allow reading "inputfile" from stdin
     implicit DBEnv  ?
     comment (not scrub) first line : in order for error messages to report correct line 

  Issues 
  ~~~~~~~

     duplicated options gives "multiple occurrences" and subsequent errors
     needs to exit earlier or handle multiple where that makes sense

  Configurable loading of indices and generic access
  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

     generic app indices by configuring queries providing (key,val) lists
     which are loaded into std::map<string,XmlValue> 

        [map.name]
        name = code2latex
        [map.query]	 
        query = for $glyph in collection('dbxml:/sys')/*[dbxml:metadata('dbxml:name')='pdgs.xml' or dbxml:metadata('dbxml:name')='extras.xml' ]//glyph return (data($glyph/@code), data($glyph/@latex)) 

     Such maps could be accessible by generic extension function my:map('code2latex',$key )


  Keeping qxml generic
  ~~~~~~~~~~~~~~~~~~~~~

     dlopen/dlsym (or C++ equivalent) handling for resolver and 
     extension functions to prevent project specifics from creeping into qxml.
     Such specifics should be being developed elsewhere (in heprez repository for example).

     Some generic extfun will be needed however, so probably best to have an umbrella
     resolver that handles
        * dynamic resolver loading
	* hands out resolve requests based on namespace namespace.

     (see env/dlfcn for tutorial of dlopen technique)
      http://www.faqs.org/docs/Linux-mini/C++-dlopen.html

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
#include "common.hh"

using namespace std;
using namespace DbXml;

typedef vector<string> svec ;
typedef map<string,string> ssmap ;
typedef map<string,ssmap> sssmap ;

/*
*/

int main(int argc, char **argv)
{
     boost::chrono::system_clock::time_point t_start, t_preload, t_postload, t_prequery, t_postquery, t_end ;
     t_start = boost::chrono::system_clock::now();

     sssmap cfg ;
     qxml_config( argc, argv, cfg );
       
     string loglevel( cfg["cli"]["level"] );  // TODO: find C++ logging approach 
     string outXml(   cfg["cli"]["outxml"] ); 
     string xqpath( cfg["cli"]["inputfile"] );
     ifstream t(xqpath.c_str()); 
     char c = t.peek();
     if(c == '#') t.ignore( numeric_limits<streamsize>::max(), '\n' );  // ignore 1st line when 1st char is '#' allowing shebang running  
     string q((istreambuf_iterator<char>(t)), istreambuf_iterator<char>());

     string envdir = cfg["dbxml"]["dbxml.environment_dir"] ;  
     prepare_dir( envdir );


     DB_ENV* env = NULL;
     int dberr = db_env_create(&env, 0);
     if (dberr) {
	  cerr << "Unable to create environment: " << db_strerror(dberr) << endl;
          if (env) env->close(env, 0);
          return EXIT_FAILURE;
     }
     u_int32_t envCacheGB    = 0 ;
     u_int32_t envCacheBytes = 64*1024*1024;   // 64 MB
     int ncache = 1 ;  
     env->set_cachesize(env, envCacheGB, envCacheBytes, ncache);

     u_int32_t env_flags = DB_CREATE |            // create environment if non-existing 
	                   DB_INIT_MPOOL  ;       // initialize the cache  

     env->open(env, envdir.c_str() , env_flags, 0);

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

	    XmlContainerConfig cconfig;
            cconfig.setAllowCreate(true);    // If the container does not exist, create it.

	    XmlContainer* cont ;
            if(chk == 0){
                cont = new XmlContainer(mgr.createContainer(name, cconfig));
	    } else {	    
		cont = new XmlContainer(mgr.openContainer(name, cconfig));   
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

      
        t_preload = boost::chrono::system_clock::now();
	// TODO: move this heprez specific elsewhere 
	resolver.readGlyphs(mgr);  
	//resolver.dumpGlyphs();
	
        t_postload = boost::chrono::system_clock::now();

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

        if (outXml == "" ){

	    // this is the only **cout**, to make it possible to output valid XML
	    XmlValue value;
            while (res.next(value)) cout << value.asString() << endl;

        } else {

             string outContainerTag = "tmp" ;		 
             ssmap::const_iterator it = cfg["containers"].find(outContainerTag) ;
             if( it == cfg["containers"].end() ){
		 cerr << "-o --outxml option requires container with alias " << outContainerTag << " to be configured " << endl ;      
             } else { 
                 string outContainerName = it->second ;  
                 clog <<  "output to container " << outContainerName << " outXml " << outXml << " looked up via alias " << outContainerTag << endl ;
	         XmlContainer outContainer = mgr.openContainer(outContainerName) ;  
	         XmlUpdateContext uc = mgr.createUpdateContext();
                 if(existsDoc(outXml, outContainer)){   
	            outContainer.deleteDocument(outXml, uc);	
 	         }
	         XmlValue outValue;
                 res.next(outValue);   // single top node assumption
                 //clog << outValue.asString() << endl ;
	         XmlEventReader& outRdr = outValue.asEventReader(); 
                 XmlDocument outDoc = mgr.createDocument();
                 outDoc.setName(outXml);
                 outDoc.setContentAsEventReader( outRdr );
	         outContainer.putDocument( outDoc , uc );   
	     }	
       } 

    } catch (XmlException &e) {
         cerr << "Exception: " << e.what() << std::endl;
    }
    t_end = boost::chrono::system_clock::now();

    boost::chrono::duration<double> d_init   = t_preload - t_start ;      
    boost::chrono::duration<double> d_load   = t_postload - t_preload ;
    boost::chrono::duration<double> d_qprep  = t_prequery - t_postload ;
    boost::chrono::duration<double> d_query  = t_postquery - t_prequery ;
    boost::chrono::duration<double> d_output = t_end - t_postquery ;
    boost::chrono::duration<double> d_total  = t_end - t_start ;

    clog << "init   " << d_init.count()   << " s (read config and setup containers)\n";
    clog << "load   " << d_load.count()   << " s (load maps)\n";
    clog << "qprep  " << d_qprep.count()  << " s (prepare query context)\n";
    clog << "query  " << d_query.count()  << " s (execute query)\n";
    clog << "output " << d_output.count() << " s (write results)\n";
    clog << "TOTAL  " << d_total.count()  << " s \n";

    return 0;
}
