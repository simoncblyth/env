%include "exception.i"
%include "typemaps.i"

%{
#include "db.h"
#include "extfun.hh"
#include <string>
using namespace DbXml;
%}

%include "std_string.i"


%module pyextfun

%{
#include "throwPyUserException.inc"
#include "makeXmlException.inc"
%}

class XmlExternalFunction;
class MyExternalFunctionPow;
class MyExternalFunctionSqrt;


class XmlExternalFunction
{
protected:
	XmlExternalFunction() {}
public:
	virtual ~XmlExternalFunction() {}
	virtual XmlResults execute(XmlTransaction &txn, XmlManager &mgr, const XmlArguments &args) const = 0;
	virtual void close() = 0;
};

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



%exception {
	try {
		$action
	} catch (XmlException &e) {
                std::cerr << "exception caught:" << std::endl << e.what() << std::endl;
		return NULL;
	}
}



