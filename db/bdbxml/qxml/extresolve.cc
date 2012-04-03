
#include "extresolve.hh"
#include <math.h>
#include "extfun.hh"

#include <boost/algorithm/string.hpp>
#include <boost/filesystem/operations.hpp>

#include <vector>
#include <string>
#include <iostream>

using namespace DbXml;
using namespace std;

typedef vector<string> svec ;


MyResolver::MyResolver() :
   _uri("http://my"),
   _xqmPath("")
{
   cout << "MyResolver ctor " << endl ; 	
}

XmlExternalFunction* MyResolver::resolveExternalFunction(XmlTransaction *txn, XmlManager &mgr,
		const std::string &uri, const std::string &name, size_t numberOfArgs) const 
{
      cout << "MyResolver resolveExternalFunction" << endl ; 	
	XmlExternalFunction *fun = 0;
        if( uri != _uri ) return fun ;

        if( numberOfArgs == 1 ){
	    
	    if(     name == "sqrt")          fun = new MyExternalFunctionSqrt();
            else if(name == "quote2values" ) fun = new QuoteToValues();

	} 
	else if ( numberOfArgs == 2 )
	{
            if(      name == "pow")          fun = new MyExternalFunctionPow();
	}	

	return fun;
}



void MyResolver::setXqmPath( const std::string xqmPath )
{
    cout << "MyResolver setXqmPath " << xqmPath << endl ; 	
    _xqmPath = xqmPath ;
}


XmlInputStream* MyResolver::resolveEntity( XmlTransaction *txn, XmlManager &mgr, const std::string &systemId,
		        const std::string &publicId ) const
{

    cout << "MyResolver resolveEntity " << systemId << ":" << publicId << endl ; 	
    string path = findEntity( systemId, publicId );
    cout << "found path " << path << endl ;
    return path.empty() ? NULL : mgr.createLocalFileInputStream(path) ;
}


std::string MyResolver::findEntity( const std::string &systemId, const std::string &publicId ) const
{
     svec dirs ;
     boost::split(dirs, _xqmPath, boost::is_any_of(":"));
     svec::const_iterator it ;
     for( it = dirs.begin() ; it != dirs.end() ; ++it ){
         string path = string(*it) + "/" + systemId ; 
         cout << "findEntity " << *it << " check " << path << endl ;   
         if(boost::filesystem::exists(path)) return path ;
     }
     return string("");
}





