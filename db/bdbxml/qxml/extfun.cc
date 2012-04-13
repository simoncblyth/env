
#include "extfun.hh"
#include "extresolve.hh"
#include "common.hh"

#include <math.h>
#include <string>
#include <map>

#include "model.hh"
#include "element.hh"

using namespace DbXml;
using namespace std;

//typedef map<string,string> ssmap ;


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


XmlResults MetaData::execute(XmlTransaction &txn, XmlManager &mgr, const XmlArguments &args) const  // my:metadata
{
	XmlResults arg0 = args.getArgument(0);
	XmlValue val0;
	arg0.next(val0);
	
        const XmlDocument& doc = val0.asDocument();
	//cout << "MetaData: asString   " << val0.asString() << endl ;
        //cout << "MetaData: asDocument " << doc << " " << doc.getName() << endl ;	

        string md_uri ;
        string md_name ;
        XmlValue md_value ;

        XmlMetaDataIterator mdi = doc.getMetaDataIterator();
        while(mdi.next(md_uri,md_name,md_value)) clog << md_uri << " " << md_name << " " << md_value.asString() << endl ;   

        double dummy = 42. ;
	XmlResults results = mgr.createResults();
	XmlValue va(dummy);
	results.add(va);
	return results;
}



XmlResults MMetaData::execute(XmlTransaction &txn, XmlManager &mgr, const XmlArguments &args) const // my:mmetadata
{
	// these arguments are the slots of the function signature
	//
        int nargs = args.getNumberOfArgs();  
        clog << "MMetaData nargs " << nargs << endl ;

	XmlResults arg0 = args.getArgument(0);
	XmlValue val0;

        string tmpContainer = _tmpContainer ;
	string tmpName      = _tmpName ;  
	XmlContainer cont = mgr.openContainer(tmpContainer) ;  
	XmlUpdateContext uc = mgr.createUpdateContext();

        // could just return the prior doc as a simple cache 
	// need to somehow key the args into digest filename 
	//
        if(existsDoc(tmpName, cont)){   
	    cont.deleteDocument(tmpName, uc);	
	    clog << "MMetaData delete and re-create " << tmpName << endl ;
	} else {    
	    clog << "MMetaData creating " << tmpName << endl ;
	}

        XmlDocument doc = mgr.createDocument();
        doc.setName(tmpName);

        // writing to the scratch document

        const unsigned char* root  = (const unsigned char*)"metadata" ;
        const unsigned char* entry = (const unsigned char*)"doc" ;

        XmlEventWriter& writer = cont.putDocumentAsEventWriter(doc, uc);  
        writer.writeStartDocument(NULL, NULL, NULL); // no XML decl
        writer.writeStartElement(root, NULL, NULL, 0, false);

        string md_uri ;
        string md_name ;
        string md_valuestring ;
        XmlValue md_value ;

	// iterating over slot0 node() sequence 
	while(arg0.hasNext()){
	    arg0.next(val0);
            const XmlDocument& doc = val0.asDocument();
            string docname = doc.getName();
            clog << "MMetaData: doc " << doc << " " << docname << endl ;	

	    int natt = 0 ;
            XmlMetaDataIterator mdi = doc.getMetaDataIterator();
            while(mdi.next(md_uri,md_name,md_value)) ++natt ;
            mdi.reset();

            writer.writeStartElement(entry, NULL, NULL, natt, false);
            while(mdi.next(md_uri,md_name,md_value)){
	        md_valuestring = md_value.asString() ; 
	        const unsigned char* attv = (const unsigned char*)md_valuestring.c_str() ; 
	        const unsigned char* attn = (const unsigned char*)md_name.c_str() ; 
                writer.writeAttribute(attn, NULL, NULL, attv , true);
		//clog << md_uri << " " << md_name << " " << md_value.asString() << endl ;   
            }		 

	    const unsigned char* text = (const unsigned char*)docname.c_str();
	    int tlen = docname.size();    // character count, excluding termination 
            writer.writeText(XmlEventReader::Characters, text, tlen);  
            writer.writeEndElement(entry, NULL, NULL);
        }

        writer.writeEndElement(root, NULL, NULL);
        writer.writeEndDocument();
        writer.close(); 

	XmlValue vret(doc);
	XmlResults ret = mgr.createResults();
	ret.add(vret);

	return ret ;
}


XmlResults QuoteToValues::execute(XmlTransaction &txn, XmlManager &mgr, const XmlArguments &args) const // my:quote2values
{
	XmlResults arg0 = args.getArgument(0);
	XmlValue val0;
	arg0.next(val0);

        Element e;   // hmm maybe this should be RezReader
        Quote   q;
	e.read_quote( q, val0 );
        q.dump();

        double dummy = 42. ;
	XmlResults results = mgr.createResults();
	XmlValue va(dummy);
	results.add(va);
	return results;
}

XmlResults CodeToLatex::execute(XmlTransaction &txn, XmlManager &mgr, const XmlArguments &args) const // my:code2latex
{
	XmlResults ret = mgr.createResults();
	XmlResults arg0 = args.getArgument(0);
	XmlValue val0;
	while(arg0.hasNext()){
	    arg0.next(val0);
	    string code = val0.asString() ;     
	    const string& latex = _resolver->codeToLatex(code);  
	    //clog << "lookup code " << code << " ==> " << latex << endl ;
	    XmlValue vret(latex);
	    ret.add(vret);
        }
	return ret ;
}



XmlResults Map::execute(XmlTransaction &txn, XmlManager &mgr, const XmlArguments &args) const // my:map
{
	XmlResults ret = mgr.createResults();

	// 1st slot for map name  
	XmlResults arg0 = args.getArgument(0);
	XmlValue val0;
        arg0.next(val0);
	string mapk = val0.asString();

	// 2nd slot for key(s) to lookup 
	XmlResults arg1 = args.getArgument(1);
	XmlValue val1;
	while(arg1.hasNext()){
	    arg1.next(val1);
	    string key = val1.asString() ;     
	    const string& val = _resolver->mapLookup( mapk , key );  
	    clog << "lookup mapk " << mapk << " key " << key << " ==> " << val << endl ;
	    XmlValue vret(val);
	    ret.add(vret);
        }
	return ret ;
}


