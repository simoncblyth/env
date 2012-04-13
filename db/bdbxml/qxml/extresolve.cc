
#include "extresolve.hh"
#include <math.h>
#include "extfun.hh"

#include <boost/algorithm/string.hpp>
#include <boost/filesystem/operations.hpp>

#include <vector>
#include <string>
#include <iostream>
#include <map>

using namespace DbXml;
using namespace std;

typedef vector<string> svec ;


MyResolver::MyResolver() :
   _uri("http://my"),
   _xqmPath(""),
   _tmpContainer(""),
   _tmpName("")
{
   //cout << "MyResolver ctor " << endl ; 	
}

XmlExternalFunction* MyResolver::resolveExternalFunction(XmlTransaction *txn, XmlManager &mgr,
		const std::string &uri, const std::string &name, size_t numberOfArgs) const 
{
        clog << "MyResolver resolveExternalFunction" << endl ; 	
	XmlExternalFunction *fun = 0;
        if( uri != _uri ) return fun ;
        if( numberOfArgs == 1 ){
	    
	    if(       name == "sqrt"){          return new MyExternalFunctionSqrt(); 
	    } else if(name == "quote2values" ){ return new QuoteToValues(); 
	    } else if(name == "metadata" )    { return new MetaData(); 
	    } else if(name == "code2latex" ){  
		CodeToLatex* c2l = new CodeToLatex();
                c2l->_resolver = this ;  
                return c2l ;

	    } else if(name == "mmetadata" ){

		MMetaData* mmd = new MMetaData();
		mmd->_tmpContainer = _tmpContainer ;
		mmd->_tmpName      = _tmpName ;
                return mmd ;
	    }
	} 
	else if ( numberOfArgs == 2 )
	{
            if(         name == "map"){ 
		Map* map = new Map();
	        map->_resolver = this ;	
	        return map ;
	    } else if(  name == "pow"){ 
		return new MyExternalFunctionPow(); 
	    }
	}	
	return fun;
}



void MyResolver::setXqmPath( const std::string xqmPath )
{
    _xqmPath = xqmPath ;
}
void MyResolver::setTmpContainer( const std::string tmpContainer )
{
    _tmpContainer = tmpContainer  ;
}
void MyResolver::setTmpName( const std::string tmpName )
{
    _tmpName = tmpName ;
}




string MyResolver::codeToLatex( const std::string& code ) const
{
    ssmap::const_iterator it = _glyph.find(code) ;
    string latex = ( it != _glyph.end()) ? it->second : "" ; 
    return latex ;
}

void MyResolver::dumpGlyphs()
{
   ssmap::const_iterator it ;
   for( it = _glyph.begin() ; it != _glyph.end() ; ++it ) clog << "    " << it->first << " : " << it->second << endl ;   
}


void MyResolver::readGlyphs( XmlManager& mgr )
{
    string q = "collection('dbxml:/sys')/*[dbxml:metadata('dbxml:name')='pdgs.xml' or dbxml:metadata('dbxml:name')='extras.xml' ]//glyph";
    XmlQueryContext qctx = mgr.createQueryContext();
    XmlResults res = mgr.query( q , qctx );
    XmlValue glyph;
    while (res.next(glyph)){
	//cout << glyph.asString() << endl;
	string code = "" ;
	string latex = "" ;
        XmlValue att ;
	XmlResults atts = glyph.getAttributes();
        while (atts.next(att)){
	    string node = att.getNodeName();
	    string value = att.getNodeValue();
	    if( node == "code" ){ 
		 code = value ;
	    } else if( node == "latex" ){ 
		 latex = value ;
		 break ;          // assuming stable attribute ordering 
            }		  
	}
        if(code.empty() || latex.empty()){
            cerr << "glyph unexpected " << endl ;
        } else { 
            _glyph[code] = latex ;
        }
    }
    clog << "readGlyphs read " << _glyph.size() << " code => latex pairs " << endl;
}


string MyResolver::mapLookup( const std::string& mapk, const std::string& key ) const
{
    string val = "" ;	
    sssmap::const_iterator imk = _map.find( mapk );
    if( imk == _map.end() ){
	cerr << "no map with key " << mapk << endl ;
    } else {
        const ssmap& kv = imk->second ;
        ssmap::const_iterator ikv = kv.find(key) ;
        if( ikv == kv.end() ){
	   cerr << "map " << mapk << " has no such key " << key << endl ;
	} else {
           val = ikv->second ;   
	}
    }	    
    return val ;
}

void MyResolver::loadMaps( XmlManager& mgr, ssmap& maps )
{
    ssmap::const_iterator it ;
    for( it = maps.begin() ; it != maps.end() ; ++it ){
        clog << "loadMaps " << it->first << " : " << it->second << endl ;

        XmlQueryContext qctx = mgr.createQueryContext();
        XmlResults res = mgr.query( it->second , qctx );
 	// alternating keys and values, by convention 
        XmlValue key;         
        XmlValue val;  
        while (res.next(key)){
            string k = key.asString();
	    string v = "" ;                // more generally would be good to grab the XmlValue here for fragment storage eg svg elements 
	    if(res.next(val)){
               v = val.asString() ;
	    }
	    if( k.empty() || v.empty() ){
		cerr << "unexpected kv " << k << " : " << v << endl ;    
	    } else { 	    
	        _map[it->first][k] = v ;
	    }
	}    
        clog << "loadMaps " << it->first << " read " << _map[it->first].size() << " kv pairs " << endl ;
    }
}




XmlInputStream* MyResolver::resolveEntity( XmlTransaction *txn, XmlManager &mgr, const std::string &systemId,
		        const std::string &publicId ) const
{
    string path = findEntity( systemId, publicId );
    clog << "MyResolver::resolveEntity (" << systemId << "{" << publicId << "}) => " << path << endl ; 	
    return path.empty() ? NULL : mgr.createLocalFileInputStream(path) ;
}


std::string MyResolver::findEntity( const std::string &systemId, const std::string &publicId ) const
{
     svec dirs ;
     boost::split(dirs, _xqmPath, boost::is_any_of(":"));
     svec::const_iterator it ;
     for( it = dirs.begin() ; it != dirs.end() ; ++it ){
         string path = string(*it) + "/" + systemId ; 
         //cout << "findEntity " << *it << " check " << path << endl ;   
         if(boost::filesystem::exists(path)) return path ;
     }
     return string("");
}





