#include "common.hh"
#include <string>
#include "dbxml/DbXml.hpp"

using namespace std;
using namespace DbXml;

// from FAQ
bool existsDoc(const string& docname, XmlContainer& cont) {
	bool ret;
	try {
		XmlDocument doc = cont.getDocument(docname, DBXML_LAZY_DOCS);
		ret = true;
	} catch (XmlException &e) {
		if (e.getExceptionCode() == XmlException::DOCUMENT_NOT_FOUND)
		    ret = false;
	        else
		    throw;   // unknown error
        }     
	return ret;
}


