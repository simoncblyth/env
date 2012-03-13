
#include "extresolve.hh"
#include <math.h>

#include "extfun.hh"

using namespace DbXml;
using namespace std;


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


