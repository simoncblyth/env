
#include <iostream>
#include <string>

#include "quote.hh"
#include "dbxml/DbXml.hpp"

using namespace std ;
using namespace DbXml ;


Quote::Quote(XmlEventReader& rdr)
{
   const char* blnk = "" ;	
   string posn("");
   string curr("");
   while (rdr.hasNext()) {
	XmlEventReader::XmlEventType type = rdr.next();
	if (type == XmlEventReader::StartElement) 
	{
            if (!rdr.isEmptyElement()){
                const unsigned char* name = rdr.getLocalName();
		curr = reinterpret_cast<const char*>(name) ;
		if( curr == "value" || curr == "err" || curr == "xerr" ) posn = curr ;
            }		    
        } 
	else if ( type == XmlEventReader::Characters )
	{
	    size_t len ;	
            const unsigned char* value = rdr.getValue(len);
	    string text = (rdr.isWhiteSpace()) ? blnk : reinterpret_cast<const char*>(value)  ; 	    
            parse(posn, curr, text );           
	}
	else if ( type == XmlEventReader::EndElement )
	{
	    curr = blnk ;	
	}
   }	
}



void Quote::parse( string& posn , string& elem , string& text )
{
    if(posn == "") return ;
    cout << "Quote::parse " << posn << " : " << elem << " : " << text << endl;

    if(posn == "value")
    {	    
    }
    else if( posn == "err" or posn == "xerr" )
    {	    
    }
}


