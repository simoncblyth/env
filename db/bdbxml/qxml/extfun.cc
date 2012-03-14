
#include "extfun.hh"
#include <math.h>

using namespace DbXml;
using namespace std;


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






XmlResults QuoteToValues::execute(XmlTransaction &txn, XmlManager &mgr, const XmlArguments &args) const
{
	XmlResults argResult1 = args.getArgument(0);
	XmlValue arg1;
	argResult1.next(arg1);

        cout << "QTV: asString    " << arg1.asString() << endl;
        cout << "QTV: getNodeType " << arg1.getNodeType() << endl;



	double result = sqrt(arg1.asNumber());
		
	XmlResults results = mgr.createResults();
	XmlValue va(result);
	results.add(va);
	
	return results;
	
}

void QuoteToValues::close()
{
	delete this;
}


