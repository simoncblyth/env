/*
   Command line tool to facilitate XQuerying dbxml containers.
   See README.txt for issues/enhancement ideas etc..
*/

#include <boost/chrono.hpp>

#include <vector>
#include <string>
#include <map>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <streambuf>
#include <sstream>
#include <ctime>

#include "dbxml/DbXml.hpp"

#include "config.hh"
#include "extresolve.hh"
#include "common.hh"

using namespace std;
using namespace DbXml;

typedef vector<string> svec ;
typedef map<string,string> ssmap ;
typedef map<string,ssmap> sssmap ;

// read from file or stdin, shebang lines are xquery commented    
string read_query( const string& xqpath )
{
     stringstream qss ;
     ifstream fin ;
     istreambuf_iterator<char> isi ;
     istreambuf_iterator<char> eos = istreambuf_iterator<char>(); 
     if( xqpath == "-" ){
         isi = istreambuf_iterator<char>(cin); 
     } else {
         fin.open( xqpath.c_str() );
         char c = fin.peek();
         if(c == '#'){
             char shebang[256];  
        fin.getline( shebang , 256 );
             qss << "(:" << shebang << ":)" << endl ;      
         }   
         isi = istreambuf_iterator<char>(fin); 
     }
     copy( isi, eos, ostreambuf_iterator<char>(qss));
     return qss.str() ;
}


string local_timestring()
{
    //  TODO: move to boost time and local with UTC offset 
    //     '2012-04-16T14:38:11+08:00'
    //
     time_t now = time( 0 );
     char timeString[100];
     strftime(timeString, 100, "%Y-%m-%dT%H:%M:%S+08:00", localtime( &now ) );
     return string(timeString);
}

int configure_dbenv( DB_ENV*& env , const string& envdir )
{
     prepare_dir( envdir );
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
}



string cfg_lookup( sssmap& cfg , string fold, string key )
{
   string val ;
   sssmap::const_iterator fit = cfg.find( fold );
   if( fit != cfg.end() ){

       ssmap::const_iterator kit = cfg[fold].find( key );
       if( kit != cfg[fold].end() ) val = cfg[fold][key] ;          
   }  
   return val ;
}


int main(int argc, char **argv)
{
     boost::chrono::system_clock::time_point t_start, t_preload, t_postload, t_prequery, t_postquery, t_end ;
     t_start = boost::chrono::system_clock::now();

     sssmap cfg ;
     qxml_config( argc, argv, cfg );
       
     string loglevel( cfg["cli"]["level"] );  // TODO: find C++ logging approach 
     string outXml(   cfg["cli"]["outxml"] ); 
     string q = read_query( cfg["cli"]["inputfile"] );
     clog << q << endl ;                   // TODO: add option switch for this dumping
     
     DB_ENV* env = NULL;
     int dberr = configure_dbenv( env, cfg["dbxml"]["dbxml.environment_dir"]) ;  
     if (dberr) exit(dberr);

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

            // need to "leak" containers to keep them open it seems
            XmlContainer* cont ;
            if(chk == 0){
                cont = new XmlContainer(mgr.createContainer(name, cconfig));
            } else {       
                cont = new XmlContainer(mgr.openContainer(name, cconfig));   
            }                  
            cont->addAlias(tag);
            if( tag == "tmp" ){                             // special meaning for container with tag "tmp" , used for scratch space     
                resolver.setTmpContainer(name);             // dbxml uri of the scratch container
                resolver.setTmpName("tmpfragment.xml");     // default name of scratch docs : TODO: check auto-naming
            }
        }

      
        t_preload = boost::chrono::system_clock::now();
        resolver.loadMaps(mgr, cfg["maps"] );

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
            // the below is the only **cout**, to make it possible to output valid XML
            clog << endl ;      
            int count = 0 ;
            XmlValue value;
            while (res.next(value)){
               cout << value.asString() << endl;
               count += 1 ;   
            }    
            clog << endl ;      
            clog << "sequence count " << count << endl ;

        } else {

             string qxmlns = cfg_lookup( cfg, "namespaces" , "qxml" );   
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
                 XmlValue createStamp(local_timestring());      // modify ?
                 clog << "setting createStamp ns " << qxmlns << " : created : " << createStamp.asString() << endl ;  
                 outDoc.setMetaData( qxmlns , "created", createStamp ); 
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


    clog << setprecision(4)  ;  
    clog << "init   " << setw(10) << d_init.count()   << " s [read config and setup containers]\n";
    clog << "load   " << setw(10) << d_load.count()   << " s [load maps]\n";
    clog << "qprep  " << setw(10) << d_qprep.count()  << " s [prepare query context]\n";
    clog << "query  " << setw(10) << d_query.count()  << " s [execute query]\n";
    clog << "output " << setw(10) << d_output.count() << " s [write results]\n";
    clog << "TOTAL  " << setw(10) << d_total.count()  << " s \n";

    return 0;
}
