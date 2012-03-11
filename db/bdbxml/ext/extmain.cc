/*
  

*/

#include <iostream>
#include <math.h>
#include <dbxml/DbXml.hpp>

#include "extfun.hh"

using namespace DbXml;
using namespace std;

int main(int argc, char **argv)
{
	// Query that calls the external function pow() 
	// The function must be declared in the query's preamble
	string query1 = 
		"declare function my:pow($a as xs:double, $b as xs:double) as xs:double external;\nmy:pow(2,3)";
		
	// Query that calls the external function sqrt() 
	string query2 = 
		"declare function my:sqrt($a as xs:double) as xs:double external;\nmy:sqrt(16)";

	try {
		
		// Create an XmlManager
		XmlManager mgr;
		
		// Create an function resolver
		MyFunResolver resolver;

		// Register the function resolver to XmlManager
		mgr.registerResolver(resolver); 

		XmlQueryContext context = mgr.createQueryContext();
		
		// Set the prefix URI
		context.setNamespace("my", resolver.getUri());

		// The first query returns the result of pow(2,3)
		XmlResults results = mgr.query(query1, context);

		XmlValue va;
		while (results.next(va)) {
			cout << "The result of pow(2,3) is : " << va.asNumber() << endl;
		}

		// The second query returns the result of sqrt(16)
		results = mgr.query(query2, context);
		while (results.next(va)) {
			cout << "The result of sqrt(16) is : " << va.asNumber() << endl;
		}

	} catch (XmlException &xe) {
		cout << "XmlException: " << xe.what() << endl;
	} 
	return 0;
}
