#ifndef EXTRESOLVE_HH
#define EXTRESOLVE_HH

#include <string>
#include <map>
#include <iostream>
#include <dbxml/DbXml.hpp>

using namespace DbXml;
using namespace std;

typedef map<string, string> ssmap;
typedef map<string, ssmap> sssmap;

class MyResolver : public XmlResolver
{
public:
	MyResolver();

	XmlExternalFunction* resolveExternalFunction(XmlTransaction *txn, XmlManager &mgr,
		const std::string &uri, const std::string &name, size_t numberOfArgs) const; 

        XmlInputStream* resolveEntity( XmlTransaction *txn, XmlManager &mgr, const std::string &systemId,
		        const std::string &publicId ) const;

	std::string findEntity( const std::string &systemId, const std::string &publicId ) const;
	std::string getUri(){ return _uri; }

	std::string codeToLatex( const std::string& code ) const;
        void dumpGlyphs();
        void readGlyphs( XmlManager& mgr );


	std::string mapLookup( const std::string& mapn, const std::string& key ) const;
        void loadMaps( XmlManager& mgr, ssmap& maps );

	// colon delimited string with directories to look for XQuery modules, 
	// searched in order with first match used
        void setXqmPath( const std::string xqmPath );
	std::string getXqmPath(){ return _xqmPath ; }

	// dbxml uri of scratch container, used for fragment preparation
        void setTmpContainer( const std::string tmpContainer );
	std::string getTmpContainer(){ return _tmpContainer ; }

	// default name of temporary fragments created in scratch container 
        void setTmpName( const std::string tmpName );
	std::string getTmpName(){ return _tmpName ; }

	// code => latex mappings 
        ssmap _glyph ;

	// generic maps populated by kv query from config 
        sssmap _map ;

private:
	string _uri ;
	string _xqmPath ; 
	string _tmpContainer ; 
	string _tmpName ; 


};

#endif
