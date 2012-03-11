
#include "extfun.hh"

using namespace DbXml;
using namespace std;
/* External function pow() implementation */
XmlResults MyExternalFunctionPow::execute(XmlTransaction &txn, XmlManager &mgr, const XmlArguments &args) const
{
	XmlResults argResult1 = args.getArgument(0);
	XmlResults argResult2 = args.getArgument(1);

	XmlValue arg1;
	XmlValue arg2;
	
	// Retrieve argument as XmlValue 
	argResult1.next(arg1);
	argResult2.next(arg2);
	
	// Call pow() from C++ 
	double result = pow(arg1.asNumber(),arg2.asNumber());
	
	// Create an XmlResults for return
	XmlResults results = mgr.createResults();
	XmlValue va(result);
	results.add(va);
	
	return results;
}

/* 
 * MyFunResolver returns a new instance of this object for each Resolution, so
 * that instance must be deleted here 
 */
void MyExternalFunctionPow::close()
{
	delete this;
}

/* External function sqrt() implementation */
XmlResults MyExternalFunctionSqrt::execute(XmlTransaction &txn, XmlManager &mgr, const XmlArguments &args) const
{
	XmlResults argResult1 = args.getArgument(0);
	XmlValue arg1;
	argResult1.next(arg1);
	
	// Call sqrt() from C++ 
	double result = sqrt(arg1.asNumber());
		
	XmlResults results = mgr.createResults();
	XmlValue va(result);
	results.add(va);
	
	return results;
	
}

/* 
 * MyFunResolver returns a new instance of this object for each Resolution, so
 * that instance must be deleted here 
 */
void MyExternalFunctionSqrt::close()
{
	delete this;
}


MyFunResolver::MyFunResolver()
	:uri_("my://my.fun.resolver")
{
}


/* 
 * Returns a new instance of either MyExternalFunctionPow or 
 * MyExternalFuncitonSqrt if the URI, function name, and number of
 * arguments match. 
 */
XmlExternalFunction* MyFunResolver::resolveExternalFunction(XmlTransaction *txn, XmlManager &mgr,
		const std::string &uri, const std::string &name, size_t numberOfArgs) const 
{
	XmlExternalFunction *fun = 0;

	if (uri == uri_ && name == "pow" && numberOfArgs == 2 )
		fun = new MyExternalFunctionPow();
	else if (uri == uri_ && name == "sqrt" && numberOfArgs == 1)
		fun = new MyExternalFunctionSqrt();

	return fun;
}


