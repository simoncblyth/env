#ifndef EXTFUN_HH
#define EXTFUN_HH

#include <iostream>
#include <math.h>
#include <dbxml/DbXml.hpp>

using namespace DbXml;
using namespace std;

class MyExternalFunctionPow : public XmlExternalFunction
{
public:
	XmlResults execute(XmlTransaction &txn, XmlManager &mgr, const XmlArguments &args) const;
	void close();

};

class MyExternalFunctionSqrt : public XmlExternalFunction
{
public:
	XmlResults execute(XmlTransaction &txn, XmlManager &mgr, const XmlArguments &args) const;
	void close();
};

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


