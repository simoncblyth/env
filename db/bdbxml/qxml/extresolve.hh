#ifndef EXTRESOLVE_HH
#define EXTRESOLVE_HH

#include <iostream>
#include <dbxml/DbXml.hpp>

using namespace DbXml;
using namespace std;

class MyFunResolver : public XmlResolver
{
public:
	MyFunResolver();
	XmlExternalFunction *resolveExternalFunction(XmlTransaction *txn, XmlManager &mgr,
		const std::string &uri, const std::string &name, size_t numberOfArgs) const; 
	string getUri(){ return uri_; }
private:
	const string uri_;
};

#endif
