
#include "extresolve.hh"
#include <math.h>

#include "extfun.hh"

using namespace DbXml;
using namespace std;


MyFunResolver::MyFunResolver():uri_("http://my")
{
}

XmlExternalFunction* MyFunResolver::resolveExternalFunction(XmlTransaction *txn, XmlManager &mgr,
		const std::string &uri, const std::string &name, size_t numberOfArgs) const 
{
	XmlExternalFunction *fun = 0;
        if( uri != uri_ ) return fun ;

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


