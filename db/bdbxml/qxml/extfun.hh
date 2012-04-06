#ifndef EXTFUN_HH
#define EXTFUN_HH

#include <dbxml/DbXml.hpp>
#include <string>

using namespace DbXml;

/* 
 * MyFunResolver returns a new instance of this object for each Resolution, so
 * that instance must be deleted here 
 */



class MyExternalFunctionPow : public XmlExternalFunction
{
public:
	XmlResults execute(XmlTransaction &txn, XmlManager &mgr, const XmlArguments &args) const;
	void close(){ delete this ;}
};

class MyExternalFunctionSqrt : public XmlExternalFunction
{
public:
	XmlResults execute(XmlTransaction &txn, XmlManager &mgr, const XmlArguments &args) const;
	void close(){ delete this ;}
};

class QuoteToValues : public XmlExternalFunction
{
public:
	XmlResults execute(XmlTransaction &txn, XmlManager &mgr, const XmlArguments &args) const;
	void close(){ delete this ;}
};

class MetaData: public XmlExternalFunction 
{
public:
	XmlResults execute(XmlTransaction &txn, XmlManager &mgr, const XmlArguments &args) const;
	void close(){ delete this ;}
};


/*
 *  Demonstrates document creation with XmlWriter from multi-nodes input
 */
class MMetaData: public XmlExternalFunction 
{
public:
	XmlResults execute(XmlTransaction &txn, XmlManager &mgr, const XmlArguments &args) const;
	void close(){ delete this ;}

	std::string _tmpContainer ;
	std::string _tmpName ;
private:

};


#endif


