#ifndef EXTRESOLVE_HH
#define EXTRESOLVE_HH

#include <iostream>
#include <dbxml/DbXml.hpp>

using namespace DbXml;
using namespace std;

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

        void setXqmPath( const std::string xqmPath );
	std::string getXqmPath(){ return _xqmPath ; }

private:
	string _uri ;
	string _xqmPath ; 
};

#endif
