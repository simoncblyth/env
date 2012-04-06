
#include "extfun.hh"
#include <math.h>

#include "model.hh"
#include "element.hh"


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
	XmlResults arg0 = args.getArgument(0);

	XmlValue val0;
	arg0.next(val0);

	//cout << "QTV: asString    " << val0.asString() << endl;
        //const XmlDocument& doc = val0.asDocument();
        //cout << "QTV: asDocument " << doc << " " << doc.getName() << endl ;	

        /*
        XmlMetaDataIterator mdi = doc.getMetaDataIterator();
        string md_uri ;
        string md_name ;
        XmlValue md_value ;
        while(mdi.next(md_uri,md_name,md_value)){
            cout << md_uri << " " << md_name << " " << md_value.asString() << endl ;   
        }	
        */

        Element e;
        Quote   q;
	e.read( q, val0 );
        q.dump();



        double dummy = 42. ;

	XmlResults results = mgr.createResults();
	XmlValue va(dummy);
	results.add(va);
	return results;
	
}

void QuoteToValues::close()
{
	delete this;
}


